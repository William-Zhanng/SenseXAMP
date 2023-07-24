from .basemodel import META_MODELS
from .TRMfusion import MultiModalFusionModel,BaseSlfAttnModel,BaseCrossAttnModel,SlfAttnEnsembelModel
META_MODELS['MultiModalFusionModel'] = MultiModalFusionModel
META_MODELS['BaseSlfAttnModel'] = BaseSlfAttnModel
META_MODELS['BaseCrossAttnModel'] = BaseCrossAttnModel
META_MODELS['SlfAttnEnsembelModel'] = SlfAttnEnsembelModel
def build_model(cfg):
    model = META_MODELS[cfg.model['name']]
    model_kwargs = cfg.model['kwargs'] if cfg.model['kwargs'] else {}
    return model(cfg=cfg,**model_kwargs)