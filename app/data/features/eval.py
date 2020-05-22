def _get_features(self, elements):
    import features
    fmmod = features.FeaturesModule()
    result = fmmod.get_features(elements)
    return result
