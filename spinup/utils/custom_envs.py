def import_custom_envs():
    # Register custom envs
    try:
        # TODO: Read env.imports or something for this
        import gym_match_input_continuous
    except ImportError:
        pass
    import deepdrive_zero