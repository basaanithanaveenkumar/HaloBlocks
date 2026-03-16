from typing import Union
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

class BlockFactory:
    """
    A factory for creating Block instances from configuration dictionaries.

    The factory supports recursive building of nested blocks (e.g., CompositeBlocks)
    by identifying "blocks" keys in the configuration.
    """
    @staticmethod
    def create(config_or_type: Union[str, dict], **kwargs) -> Block:
        """
        Instantiates a Block based on the provided configuration.

        Supports two invocation styles:
        1. Dictionary: create({'type': 'mha', 'dim': 128})
        2. Keyword: create('mha', dim=128)

        Args:
            config_or_type (str | dict): Either a configuration dictionary
                containing "type" or the string name of the block type.
            **kwargs: Parameters passed to the block constructor if config_or_type is a string.

        Returns:
            Block: An initialized Block instance.
        """
        if isinstance(config_or_type, dict):
            config = config_or_type.copy()
            block_type = config.pop("type")
            params = config
        else:
            block_type = config_or_type
            params = kwargs

        block_cls = BlockRegistry.get(block_type)
        
        # Recursively build nested blocks if needed
        if "blocks" in params:
            sub_blocks = []
            for sub in params["blocks"]:
                if isinstance(sub, dict):
                    sub_blocks.append(BlockFactory.create(sub))
                elif isinstance(sub, (list, tuple)) and len(sub) == 2 and isinstance(sub[1], dict):
                    sub_blocks.append(BlockFactory.create(sub[0], **sub[1]))
                else:
                    sub_blocks.append(sub)
            params["blocks"] = sub_blocks
            
        return block_cls(**params)