class BlockFactory:
    @staticmethod
    def create(config: dict) -> Block:
        block_type = config.pop("type")
        block_cls = BlockRegistry.get(block_type)
        # Recursively build nested blocks if needed
        if "blocks" in config:
            # e.g., for CompositeBlock
            sub_blocks = [BlockFactory.create(sub) for sub in config["blocks"]]
            config["blocks"] = sub_blocks
        return block_cls(**config)