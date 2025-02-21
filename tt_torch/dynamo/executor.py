# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tt_torch.tools.utils import CompilerConfig
import torch


def compile_process(receiver, sender, ttir_event, ttnn_event, json_event):
    obj = receiver.get()
    faulthandler.disable()
    asm = obj["asm"]
    ttir = tt_mlir.compile_stable_hlo_to_ttir(asm)
    sender.put({"ttir": ttir})
    ttir_event.wait()
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    sender.put({"binary": binary, "ttnn": ttnn})
    ttnn_event.wait()
    sender.put({"json": tt_mlir.bytestream_to_json(binary)})
    json_event.wait()
    sys.exit(0)


def execute_process(receiver, sender, exec_event):
    while 1:
        obj = receiver.get()
        faulthandler.disable()
        binary = obj["binary"]
        inputs = obj["inputs"]
        file_name = obj["dump_file"]
        file_stderr = open(file_name, "w")
        old_stderr = sys.stderr
        sys.stderr = file_stderr
        old_stdout = sys.stdout
        sys.stdout = file_stderr
        outputs = tt_mlir.run(inputs, binary)
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        file_stderr.close()
        sender.put({"outputs": outputs})
        exec_event.wait()
    sys.exit(0)


class Executor:
    def __init__(
        self,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):

        self.binary = None
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.required_atol = required_atol
        self.required_pcc = required_pcc

        # Dictionary to keep track of the type conversion for unsupported hardware
        # types and use it to convert the input arguments to supported types.
        self.type_conversion = {
            torch.bool: torch.bfloat16,
            torch.int64: torch.int32,
            torch.float64: torch.float32,
        }

    def __call__(self, *inputs):
        if self.compiler_config.compile_depth != CompileDepth.EXECUTE:
            return self.gm(*inputs)

        assert self.binary is not None

        new_inputs = ()
        for input in inputs:
            # Handle scalar inputs.
            if not hasattr(input, "dtype"):
                assert (
                    type(input) is not bool
                ), "Conversion for scalar boolean is not supported."
                new_inputs = new_inputs + ((input),)
                continue

            # Apply type conversion if required.
            input_type = input.dtype
            if input_type in self.type_conversion.keys():
                new_inputs = new_inputs + (
                    (input.to(dtype=self.type_conversion[input_type])),
                )
                continue

            # No conversion required.
            new_inputs = new_inputs + ((input),)
        inputs = new_inputs
        return tt_mlir.run(inputs, self.binary)

    def set_binary(self, binary):
        self.binary = binary


class OpByOpExecutor(Executor):
    def __init__(
        self,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):
        super().__init__(compiler_config, required_pcc, required_atol)

    def compile_op(self, node, *inputs, **kwargs):
        input_shapes_and_constants = self.get_input_shapes_and_constants(inputs)
        if not self.is_node_valid(node):
            return None, None

        name = self.get_op_name(node)
        op = Op(name, input_shapes_and_constants, self.compiler_config.model_name)
        if op.unique_key() not in self.compiler_config.unique_ops:
            self.compiler_config.unique_ops[op.unique_key()] = op
        else:
            self.compiler_config.unique_ops[op.unique_key()].num_ops += 1
            return None, None

        module, op = self.get_stable_hlo_graph(node, inputs, **kwargs)

        sender = mp.Queue()
        receiver = mp.Queue()
        ttir_event = mp.Event()
        ttnn_event = mp.Event()
        json_event = mp.Event()
        obj = {"asm": module.operation.get_asm()}
        process = mp.Process(
            target=compile_process,
            args=(sender, receiver, ttir_event, ttnn_event, json_event),
        )
        process.start()
        sender.put(obj)
        start = time.time()
        binary = None
        while True:
            try:
                result = receiver.get_nowait()
                if "ttir" in result:
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTIR
                    op.add_ttir_graph(result["ttir"])
                    ttir_event.set()
                if "binary" in result:
                    binary = result["binary"]
                    op.binary = binary
                    op.add_ttnn_graph(result["ttnn"])
                    ttnn_event.set()
                if "json" in result:
                    op.json = result["json"]
                    json_event.set()
                    op.parse_json()
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN
                    break
            except mp.queues.Empty:
                pass
            except Exception as e:
                process.terminate()
                raise e
            if time.time() - start > self.compiler_config.single_op_timeout:
                process.terminate()
                break
            if not process.is_alive():
                break
            time.sleep(0.01)
        process.join()
        print(f"json len {len(op.json)}")
        return binary, op

    def run_op(self, binary, *inputs):
        inputs = self.pre_process_inputs(*inputs)
        if not self.stderror_redirected:
            self.file_stderr = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
            self.stderror_redirected = True

        obj = {"binary": binary, "inputs": inputs, "dump_file": self.file_stderr.name}

        exec_event = mp.Event()
        if self.execute_process is None:
            self.execute_sender = mp.Queue()
            self.execute_receiver = mp.Queue()
            self.execute_process = mp.Process(
                target=execute_process,
                args=(self.execute_sender, self.execute_receiver, exec_event),
            )
            self.execute_process.start()
        self.execute_sender.put(obj)
        result = {}
        start = time.time()
        outputs = [None]
        while True:
            if not self.execute_process.is_alive():
                self.execute_process = None
                break
            try:
                result = self.execute_receiver.get_nowait()
                outputs = result["outputs"]
                exec_event.set()
                break
            except mp.queues.Empty:
                pass
            if time.time() - start > self.compiler_config.single_op_timeout:
                self.execute_process.terminate()
                self.execute_process = None
                break

        if len(outputs) == 1:
            outputs = outputs[0]

        stderr_data = ""
        if outputs is None:
            file_stderr = open(self.file_stderr.name, "r")
            stderr_data = file_stderr.read()
            stderr_data = stderr_data.replace("\n", "\\n")
            stderr_data = re.sub(r"[^\x20-\x7E]", "", stderr_data)
            file_stderr.close()

        return outputs, stderr_data
