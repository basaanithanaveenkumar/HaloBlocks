class BlockRegistry:
    _registry = {}

    @classmethod
    def register(cls, name=None):
        def decorator(block_cls):
            key = name or block_cls.__name__
            cls._registry[key] = block_cls
            return block_cls
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry[name]