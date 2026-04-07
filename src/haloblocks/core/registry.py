class BlockRegistry:
    """
    A central registry for mapping string identifiers to Block classes.

    This registry enables configuration-driven model building by allowing
    the BlockFactory to look up components by their registered names.
    """

    _registry = {}

    @classmethod
    def register(cls, name=None):
        """
        Decorator to register a Block class.

        Args:
            name (str, optional): The name to register the block under.
                Defaults to the class name if not provided.
        """

        def decorator(block_cls):
            key = name or block_cls.__name__
            cls._registry[key] = block_cls
            return block_cls

        return decorator

    @classmethod
    def get(cls, name):
        """
        Retrieves a registered Block class by name.

        Args:
            name (str): The registered name of the block.

        Returns:
            Type[Block]: The registered Block class.

        Raises:
            KeyError: If the name is not registered.
        """
        return cls._registry[name]
