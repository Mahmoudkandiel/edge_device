import sophon.sail as sail
import numpy as np
import logging
import os

# Set logging level
logging.basicConfig(level=logging.INFO)

class SophonInference:
    def __init__(self, bmodel_path: str, device_id: int = 0, io_mode: sail.IOMode = sail.IOMode.SYSIO):
        """
        Generic Wrapper for Sophon Inference.
        
        Args:
            bmodel_path (str): Path to the .bmodel file.
            device_id (int): TPU device ID (default 0).
            io_mode (sail.IOMode): SYSIO (default) or DEVIO.
        """
        if not os.path.exists(bmodel_path):
            raise FileNotFoundError(f"Model not found at: {bmodel_path}")

        try:
            self.net = sail.Engine(bmodel_path, device_id, io_mode)
        except Exception as e:
            raise RuntimeError(f"Failed to load sail.Engine: {e}")


        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)        
        self.inputs_info = {}
        self.outputs_info = {}

        for name in self.input_names:
            shape = self.net.get_input_shape(self.graph_name, name)
            self.inputs_info[name] = {
                "shape": shape,
                "batch_size": shape[0],
                "height": shape[2] if len(shape) == 4 else None,
                "width": shape[3] if len(shape) == 4 else None
            }
            logging.info(f"Detected Input: {name} | Shape: {shape}")

        for name in self.output_names:
            shape = self.net.get_output_shape(self.graph_name, name)
            self.outputs_info[name] = {
                "shape": shape
            }
            logging.info(f"Detected Output: {name} | Shape: {shape}")

    def get_input_details(self):
        """Returns input names and shapes so external code knows how to resize."""
        return self.inputs_info

    def infer(self, input_data: dict) -> dict:
        """
        Run inference on the TPU.
        
        Args:
            input_data (dict): Key is input_name, Value is numpy array.
                               Example: {'images': np.array(...)}
        
        Returns:
            dict: Key is output_name, Value is numpy array.
        """
        
        # 1. Validation & Formatting
        formatted_inputs = {}
        
        for name, data in input_data.items():
            if name not in self.inputs_info:
                logging.warning(f"Input '{name}' provided but not found in model graph. Ignoring.")
                continue

            target_shape = self.inputs_info[name]["shape"]
            
            # 1a. Batch Padding (Safety Feature)
            # If the model expects batch=4 but you send batch=1, generic SDKs fail. 
            # We must pad it.
            if data.shape[0] < target_shape[0]:
                pad_width = target_shape[0] - data.shape[0]
                # Pad only the batch dimension (axis 0)
                # Create a tuple of padding widths: ((pad_batch, 0), (0,0), (0,0), (0,0))
                pad_config = [(0, pad_width)] + [(0, 0) for _ in range(data.ndim - 1)]
                data = np.pad(data, pad_config, mode='constant', constant_values=0)
            
            # 1b. Data Type Force (TPUs usually strict about float32 vs int)
            # Assuming float32 for most vision models, but you might make this dynamic if needed.
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            formatted_inputs[name] = data

        # 2. Execution (The "Black Box" part)
        # sail.Engine.process returns a dict of generic dictionary
        raw_outputs = self.net.process(self.graph_name, formatted_inputs)

        # 3. Post-Conversion (Sail Tensor -> Numpy)
        final_outputs = {}
        for name in self.output_names:
            if name in raw_outputs:
                # If using SYSIO, raw_outputs[name] is usually already a numpy array 
                # or a SailTensor depending on the exact SDK version.
                # The line below ensures we return a pure numpy array to your python code.
                val = raw_outputs[name]
                if isinstance(val, sail.Tensor):
                    final_outputs[name] = val.asnumpy()
                else:
                    final_outputs[name] = val
                    
        return final_outputs