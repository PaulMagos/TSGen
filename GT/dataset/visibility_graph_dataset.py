
from typing import Union, Optional, Callable, Mapping, Tuple, Any, Dict
from tsl.typing import DataArray, SparseTensArray, TemporalIndex
from tsl.data.spatiotemporal_dataset import SpatioTemporalDataset, Scaler
from tsl.data.batch_map import BatchMap
from .vector_visibility_graph import VectorVisibilityGraph

__all__ = ['VisibilityGraphDataset']

class VisibilityGraphDataset(SpatioTemporalDataset):
    def __init__(self,
                 target: DataArray,
                 use_vvg: bool = True,
                 vvg_params: Optional[Dict[str, Any]] = None,
                 index: Optional[TemporalIndex] = None,
                 mask: Optional[DataArray] = None,
                 connectivity: Optional[Union[SparseTensArray,
                                              Tuple[DataArray]]] = None,
                 covariates: Optional[Mapping[str, DataArray]] = None,
                 input_map: Optional[Union[Mapping, BatchMap]] = None,
                 target_map: Optional[Union[Mapping, BatchMap]] = None,
                 auxiliary_map: Optional[Union[Mapping, BatchMap]] = None,
                 scalers: Optional[Mapping[str, Scaler]] = None,
                 trend: Optional[DataArray] = None,
                 transform: Optional[Callable] = None,
                 window: int = 12,
                 horizon: int = 1,
                 delay: int = 0,
                 stride: int = 1,
                 window_lag: int = 1,
                 horizon_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        super().__init__(target=target,
                         index=index,
                         mask=mask,
                         connectivity=connectivity,
                         covariates=covariates,
                         input_map=input_map,
                         target_map=target_map,
                         auxiliary_map=auxiliary_map,
                         scalers=scalers,
                         trend=trend,
                         transform=transform,
                         window=window,
                         horizon=horizon,
                         delay=delay,
                         stride=stride,
                         window_lag=window_lag,
                         horizon_lag=horizon_lag,
                         precision=precision,
                         name=name)
        
        self.use_vvg = use_vvg
        self.vvg_params = vvg_params if vvg_params is not None else {}
         if self.use_vvg:
            self.vvg = VectorVisibilityGraph(**(vvg_params or {}))
            self.vvg_features = self._precompute_vvg()
        else:
            self.vvg_features = None
        self.visibility_graph = None  # Placeholder for visibility graph data