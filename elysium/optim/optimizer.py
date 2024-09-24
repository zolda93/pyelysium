from elysium.nn import Parameter,Sequential
import inspect
class Optim:
    def __init__(self,model):
        self.model = model
        self.parameters = self._collect_parameters()
    def _collect_parameters(self):
        parameters = []
        def collect_from_layer(layer, prefix=""):
            """
            Recursively collect parameters from both Sequential and normal layers.
            """
            if isinstance(layer, Sequential):
                # If it's a Sequential block, recursively handle its layers
                for idx, sub_layer in enumerate(layer.layers):
                    collect_from_layer(sub_layer, f"{prefix}Sequential_{idx}.")
            else:
                # Handle normal layers or model components
                if not inspect.isclass(layer) and hasattr(layer,'__call__'):
                    for name, value in layer.__dict__.items():
                        if isinstance(value, Parameter):
                            parameters.append((f"{prefix}{layer.__class__.__name__}.{name}", value))
        
        # Start collecting parameters from the root model
        for attr_name, attr_value in self.model.__dict__.items():
            collect_from_layer(attr_value, f"{attr_name}.")

        return parameters
    def step(self):
        raise NotImplementedError("Subclasses must implement the `step` method.")
    def zero_grad(self):
        for _,value in self.parameters:
            value.zero_grad()


