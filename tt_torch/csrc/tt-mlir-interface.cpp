// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

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

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTTraits.h"

#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "tt/runtime/runtime.h"

#include "tt_mlir_version.h"

#include <llvm/ADT/SmallVector.h>

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
  const char *enable_printing = std::getenv("TT_TORCH_IR_LOG_LEVEL");
  if (enable_printing && std::string(enable_printing) == "DEBUG") {
    shlo_pm.getContext()->disableMultithreading();
    shlo_pm.enableIRPrinting();
  }
  mlir::tt::ttir::StableHLOToTTIRPipelineOptions shlo_options;
  shlo_options.arithDialectConversionsEnabled = true;
  shlo_options.legalizeCompositeToCallEnabled = true;
  mlir::tt::ttir::createStableHLOToTTIRPipeline(shlo_pm, shlo_options);
  // Run the pass manager.
  if (mlir::failed(shlo_pm.run(mlir_module.get()))) {
    throw std::runtime_error(
        "Failed to run StableHLO to TTIR compiler pass pipeline.");
  }
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  mlir_module.get()->print(os, mlir::OpPrintingFlags().enableDebugInfo());
  os.flush();

  return buffer;
}

std::string get_system_desc_version_path(std::string_view system_desc_path) {
  return std::string(system_desc_path) + ".ttmlir_version";
}

bool is_system_desc_fresh(std::string_view system_desc_path) {
  std::string system_desc_version;

  std::ifstream version_file(get_system_desc_version_path(system_desc_path));
  std::getline(version_file, system_desc_version);
  version_file.close();

  return TT_MLIR_GIT_COMMIT == system_desc_version;
}

void create_system_desc(std::optional<tt::runtime::Device> device) {
  const char *system_desc_path = std::getenv("SYSTEM_DESC_PATH");

  if (!system_desc_path) {
    throw std::runtime_error(
        "SYSTEM_DESC_PATH environment variable not set.\n"
        "Please set a path or call `source env/activate`.");
  }

  if (!device.has_value() && std::filesystem::exists(system_desc_path) &&
      is_system_desc_fresh(system_desc_path)) {
    return;
  }

  std::remove(system_desc_path);

  bool creating_temp_device = !device.has_value();
  if (creating_temp_device) {
    device = tt::runtime::openMeshDevice({1, 1});
  }
  tt::runtime::getCurrentSystemDesc(std::nullopt, device)
      .first.store(system_desc_path);

  std::ofstream version_file(get_system_desc_version_path(system_desc_path));
  version_file << TT_MLIR_GIT_COMMIT << std::endl;
  version_file.close();

  if (creating_temp_device) {
    tt::runtime::closeMeshDevice(*device);
  }
}

std::tuple<std::shared_ptr<void> *, std::string>
compileTTIRToTTNN(std::string_view code,
                  std::optional<tt::runtime::Device> device,
                  size_t len_activations, size_t len_graph_constants) {

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
  const char *enable_printing = std::getenv("TT_TORCH_IR_LOG_LEVEL");
  if (enable_printing && std::string(enable_printing) == "DEBUG") {
    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }

  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;

  options.enableConstEval = true;
  options.enableFusing = true;

  if (len_activations > 0 || len_graph_constants > 0) {
    llvm::SmallVector<mlir::tt::ArgumentType> argTypes;
    for (size_t i = 0; i < len_graph_constants; ++i) {
      argTypes.push_back(mlir::tt::ArgumentType::Constant);
    }
    for (size_t i = 0; i < len_activations; ++i) {
      argTypes.push_back(mlir::tt::ArgumentType::Input);
    }
    llvm::StringMap<llvm::SmallVector<mlir::tt::ArgumentType>> argTypesMap;
    argTypesMap["main"] = argTypes;
    options.argumentTypeMap = argTypesMap;
  }

  if (std::filesystem::path system_desc_path = std::getenv("SYSTEM_DESC_PATH");
      !system_desc_path.empty()) {
    if (device.has_value()) {
      create_system_desc(device);
    }

    if (std::filesystem::exists(system_desc_path)) {
      std::filesystem::path system_desc_temp_path =
          std::filesystem::temp_directory_path() /
          (system_desc_path.stem().string() + "_tmp" +
           system_desc_path.extension().string());

      std::filesystem::copy_file(
          system_desc_path, system_desc_temp_path,
          std::filesystem::copy_options::overwrite_existing);

      options.systemDescPath = system_desc_temp_path;
    }
  }

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
  mlir_module->print(os, mlir::OpPrintingFlags().enableDebugInfo());
  os.flush();

  return std::make_tuple(binary, buffer);
}

std::shared_ptr<void> *Compile(std::string_view code) {
  std::string ttir = compileStableHLOToTTIR(code);
  return std::get<0>(compileTTIRToTTNN(ttir));
}

} // namespace tt::torch
