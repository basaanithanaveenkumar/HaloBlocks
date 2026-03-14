from haloblocks.core.registry import BlockRegistry

class LayersProxy:
    """
    A dynamic proxy that provides access to registered Block classes
    as if they were attributes of this module.
    """
    def __getattr__(self, name):
        try:
            return BlockRegistry.get(name)
        except KeyError:
            raise AttributeError(f"No block registered with name: {name}")

    def __dir__(self):
        return list(BlockRegistry._registry.keys())

# Export as a singleton-like instance
import sys
sys.modules[__name__] = LayersProxy()
