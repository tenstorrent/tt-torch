// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>

#include "tt-mlir-interface.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/IR/Attributes.h"        // from @llvm-project
#include "mlir/IR/Builders.h"          // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h" // from @llvm-project
#include "mlir/IR/BuiltinOps.h"        // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"      // from @llvm-project
#include "mlir/IR/MLIRContext.h"       // from @llvm-project
#include "mlir/IR/OwningOpRef.h"       // from @llvm-project
#include "mlir/IR/Visitors.h"          // from @llvm-project
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"              // from @llvm-project
#include "mlir/Pass/PassManager.h"           // from @llvm-project
#include "mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/Support/LogicalResult.h"      // from @llvm-project
#include "mlir/Transforms/Passes.h"          // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"       // from @stablehlo
#include "stablehlo/dialect/Register.h"      // from @stablehlo
#include "stablehlo/dialect/Serialization.h" // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"     // from @stablehlo

#define TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "tt/runtime/runtime.h"

namespace tt::torch {

std::string compileStableHLOToTTIR(std::string_view code) {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;

  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();

  mlir::tt::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::tt::registerAllExtensions(registry);

  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(code.data(), code.size()),
          // IR may be invalid because some fields may be using DenseElements
          // instead of DenseArray. We rectify that below and verify after.
          mlir::ParserConfig{&context, /*verifyAfterParse=*/true});

  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerPasses();

  // Implicit nesting required to call the stablehlo.composite --> func.call
  // conversion.
  mlir::PassManager shlo_pm(mlir_module.get()->getName(),
                            mlir::PassManager::Nesting::Implicit);
  mlir::tt::ttir::StableHLOToTTIRPipelineOptions shlo_options;
  shlo_options.arithDialectConversionsEnabled = true;
  shlo_options.removeDeadValuesEnabled = true;
  shlo_options.legalizeCompositeToCallEnabled = true;
  mlir::tt::ttir::createStableHLOToTTIRPipeline(shlo_pm, shlo_options);
  // Run the pass manager.
  if (mlir::failed(shlo_pm.run(mlir_module.get()))) {
    throw std::runtime_error(
        "Failed to run StableHLO to TTIR compiler pass pipeline.");
  }
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  mlir_module.get()->print(os);
  os.flush();

  return buffer;
}

std::tuple<std::shared_ptr<void> *, std::string>
compileTTIRToTTNN(std::string_view code) {

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;

  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();

  mlir::tt::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::tt::registerAllExtensions(registry);

  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(code.data(), code.size()),
          // IR may be invalid because some fields may be using DenseElements
          // instead of DenseArray. We rectify that below and verify after.
          mlir::ParserConfig{&context, /*verifyAfterParse=*/true});

  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerPasses();

  mlir::PassManager pm(mlir_module.get()->getName());
  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(pm, options);

  // Run the pass manager.
  if (mlir::failed(pm.run(mlir_module.get()))) {
    throw std::runtime_error(
        "Failed to run TTIR TO TTNN compiler pass pipeline.");
  }

  std::shared_ptr<void> *binary = new std::shared_ptr<void>();
  *binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());

  if (binary == nullptr) {
    throw std::runtime_error("Failed to generate flatbuffer binary.");
  }

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  mlir_module->print(os);
  os.flush();

  return std::make_tuple(binary, buffer);
}

std::shared_ptr<void> *Compile(std::string_view code) {

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;

  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();

  mlir::tt::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::tt::registerAllExtensions(registry);

  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(code.data(), code.size()),
          // IR may be invalid because some fields may be using DenseElements
          // instead of DenseArray. We rectify that below and verify after.
          mlir::ParserConfig{&context, /*verifyAfterParse=*/true});

  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerPasses();

  // Implicit nesting required to call the stablehlo.composite --> func.call
  // conversion.
  mlir::PassManager shlo_pm(mlir_module.get()->getName(),
                            mlir::PassManager::Nesting::Implicit);
  mlir::tt::ttir::StableHLOToTTIRPipelineOptions shlo_options;
  shlo_options.arithDialectConversionsEnabled = true;
  shlo_options.removeDeadValuesEnabled = true;
  shlo_options.legalizeCompositeToCallEnabled = true;
  mlir::tt::ttir::createStableHLOToTTIRPipeline(shlo_pm, shlo_options);
  // Run the pass manager.
  if (mlir::failed(shlo_pm.run(mlir_module.get()))) {
    throw std::runtime_error(
        "Failed to run StableHLO to TTIR compiler pass pipeline.");
  }

  mlir::PassManager pm(mlir_module.get()->getName());
  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(pm, options);

  // Run the pass manager.
  if (mlir::failed(pm.run(mlir_module.get()))) {
    throw std::runtime_error(
        "Failed to run TTIR TO TTNN compiler pass pipeline.");
  }

  std::shared_ptr<void> *binary = new std::shared_ptr<void>();
  *binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());

  if (binary == nullptr) {
    throw std::runtime_error("Failed to generate flatbuffer binary.");
  }

  return binary;
}

} // namespace tt::torch
