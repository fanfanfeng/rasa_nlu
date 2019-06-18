def configure_colored_logging(loglevel):
    import coloredlogs
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles['asctime'] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles['debug'] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt='%(asctime)s %(levelname)-8s %(name)s  - %(message)s',
        level_styles=level_styles,
        field_styles=field_styles)


def convert_to_dict(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value) and not name.startswith('_'):
            pr[name] = value
    return pr
