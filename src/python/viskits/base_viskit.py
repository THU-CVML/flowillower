# flowillower/viskits/base_viskit.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type, Any, Optional, Callable, List, TypeVar
import streamlit as st
import json 
import tomli
import tomli_w
from dataclasses import is_dataclass 

# Try importing Pydantic and streamlit-pydantic
PYDANTIC_AVAILABLE = False
STREAMLIT_PYDANTIC_AVAILABLE = False
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
    try:
        import streamlit_pydantic as sp
        STREAMLIT_PYDANTIC_AVAILABLE = True
    except ImportError:
        print("提示: streamlit-pydantic 未安装。配置UI将需要手动实现或使用简化的默认UI。")
        print("Suggestion: pip install streamlit-pydantic")
except ImportError:
    print("提示: Pydantic V2 未安装。基于Pydantic的配置模型将不可用。")
    print("Suggestion: pip install pydantic~=2.0")

ConfigPydanticModelType = TypeVar('ConfigPydanticModelType', bound='BaseModel' if PYDANTIC_AVAILABLE else Any)

VISKIT_REGISTRY: Dict[str, Type["VisKit"]] = {} 

def register_viskit(name: str): 
    def decorator(cls: Type["VisKit"]): 
        if name in VISKIT_REGISTRY:
            print(f"警告: 视件 '{name}' 已被注册，将被覆盖。") 
        VISKIT_REGISTRY[name] = cls
        return cls
    return decorator

def get_viskit_class(type_name: str) -> Optional[Type["VisKit"]]: 
    return VISKIT_REGISTRY.get(type_name)

class VisKit(ABC): 
    ui_config: Any 
    LOGICAL_DATA_SOURCE_NAME = "default_data_source" # Default logical name for primary data
    COLLECTION_DATA_TYPE_FOR_IDE = "generic_ide_collection_v1" # Default collection type for IDE if many assets

    def __init__(self,
                 instance_id: str, 
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]], 
                 specific_ui_config_dict: Optional[Dict[str, Any]] = None 
                ):
        self.instance_id = instance_id 
        self.trial_root_path = trial_root_path
        self.data_sources_map = data_sources_map
        
        self.private_storage_path = self.trial_root_path / "viskits_data" / self.instance_id 
        self.private_storage_path.mkdir(parents=True, exist_ok=True)

        ConfigModelClass = self.get_config_model()
        initial_values = {}
        if ConfigModelClass and PYDANTIC_AVAILABLE and issubclass(ConfigModelClass, BaseModel):
            try:
                temp_config_instance = ConfigModelClass()
                initial_values = temp_config_instance.model_dump()
            except Exception as e:
                st.error(f"视件 {self.instance_id}: 初始化默认Pydantic配置失败: {e}")
        
        loaded_config_dict = self._load_config_dict_from_file()
        if loaded_config_dict:
            initial_values.update(loaded_config_dict) 

        if specific_ui_config_dict: 
            initial_values.update(specific_ui_config_dict)

        if ConfigModelClass and PYDANTIC_AVAILABLE and issubclass(ConfigModelClass, BaseModel):
            try:
                self.ui_config = ConfigModelClass(**initial_values)
            except ValidationError as ve:
                st.error(f"视件 {self.instance_id}: UI配置验证失败。将使用默认值。\n{ve}")
                try: self.ui_config = ConfigModelClass() 
                except Exception as e_fallback:
                    st.error(f"视件 {self.instance_id}: 回退到默认Pydantic配置也失败: {e_fallback}")
                    self.ui_config = {} 
            except Exception as e: 
                 st.error(f"视件 {self.instance_id}: 创建Pydantic UI配置实例失败: {e}")
                 self.ui_config = {}
        elif ConfigModelClass and is_dataclass(ConfigModelClass): 
            try: self.ui_config = ConfigModelClass(**initial_values)
            except Exception as e:
                st.error(f"视件 {self.instance_id}: 创建dataclass UI配置实例失败: {e}")
                self.ui_config = {}
        else: 
            self.ui_config = initial_values if initial_values else {}

        self._current_global_step: Optional[int] = None
        self._on_global_step_change_request: Optional[Callable[[int], None]] = None
        self._all_available_steps: Optional[List[int]] = None 

    @classmethod
    @abstractmethod
    def get_config_model(cls) -> Optional[Type[ConfigPydanticModelType]]: 
        pass
    
    @classmethod
    def get_default_config_as_dict(cls) -> Dict[str, Any]:
        ConfigModel = cls.get_config_model()
        if ConfigModel and PYDANTIC_AVAILABLE and issubclass(ConfigModel, BaseModel):
            try: return ConfigModel().model_dump(mode='json') 
            except Exception as e:
                print(f"错误：无法为 {cls.__name__} 创建默认Pydantic配置字典: {e}")
                return {}
        elif ConfigModel and is_dataclass(ConfigModel): 
             try:
                from dataclasses import asdict
                return asdict(ConfigModel())
             except Exception as e:
                print(f"错误：无法为 {cls.__name__} 创建默认dataclass配置字典: {e}")
                return {}
        return {}

    @abstractmethod
    def report_data(self,
                    data_payload: Any,
                    step: int,
                    group_id: Optional[str] = None, 
                    asset_name: Optional[str] = None, 
                    **kwargs) -> List[Dict[str, Any]]:
        pass 


    def _get_data_asset_info(self, logical_name: str = "default") -> Optional[Dict[str, Any]]:
        # If logical_name is not provided, use the class's default logical name
        if logical_name == "default":
            logical_name_to_use = getattr(self, 'LOGICAL_DATA_SOURCE_NAME', 'default')
        else:
            logical_name_to_use = logical_name
        return self.data_sources_map.get(logical_name_to_use)


    def _get_data_asset_path(self, logical_name: str = "default") -> Optional[Path]:
        asset_info = self._get_data_asset_info(logical_name) # Uses the potentially overridden logical_name
        if asset_info and "path" in asset_info:
            path_str_or_obj = asset_info["path"]
            if not isinstance(path_str_or_obj, (str, Path)):
                st.error(f"数据资产 '{logical_name}' 的路径类型无效: {type(path_str_or_obj)}")
                return None
            return (self.trial_root_path / path_str_or_obj).resolve()
        return None

    def configure_global_step_interaction(self,
                                        current_step: Optional[int],
                                        all_available_steps: Optional[List[int]],
                                        on_step_change_request_callback: Optional[Callable[[int], None]]):
        self._current_global_step = current_step
        self._all_available_steps = sorted(list(set(all_available_steps))) if all_available_steps else []
        self._on_global_step_change_request = on_step_change_request_callback

    def _request_global_step_change(self, new_step: int) -> None:
        if self._on_global_step_change_request:
            self._on_global_step_change_request(new_step)
        else:
            st.warning(f"视件 {self.instance_id}: 尝试更改全局步骤，但未设置回调。")

    def _get_closest_available_step(self, target_step: Optional[int]) -> Optional[int]:
        if target_step is None:
            return self._all_available_steps[-1] if self._all_available_steps else None
        if not self._all_available_steps:
            return None
        if target_step in self._all_available_steps:
            return target_step
        try:
            closest = min(self._all_available_steps, key=lambda x: abs(x - target_step))
            return closest
        except ValueError: 
            return None

    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        pass

    @classmethod
    def get_display_name(cls) -> str: 
        name = cls.__name__
        if name.endswith("VisKit"): 
            name = name[:-len("VisKit")]
        elif name.endswith("Visualizer"): 
             name = name[:-len("Visualizer")]
        
        for reg_name, comp_cls in VISKIT_REGISTRY.items(): 
            if comp_cls == cls:
                name = reg_name 
                break
        
        import re
        name = name.replace("_", " ")
        name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        return name.title()

    @classmethod
    @abstractmethod
    def _generate_example_payloads_and_steps(cls, 
                                             data_sources_config: Optional[Dict[str, Any]] = None
                                             ) -> List[Dict[str, Any]]:
        # Subclasses must implement this to provide a list of dicts,
        # each dict being the kwargs for a single call to report_data.
        # Example: [{"data_payload": ..., "step": ...}, {"data_payload": ..., "step": ...}]
        return [] 
        
    @classmethod
    def generate_example_data(cls,
                              ide_instance_id: str, 
                              ide_trial_root_path: Path, 
                              data_sources_config: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Dict[str, Any]]:
        temp_vis_instance = cls( 
            instance_id=ide_instance_id, 
            trial_root_path=ide_trial_root_path,   
            data_sources_map={},                   
            specific_ui_config_dict=None    
        )

        report_calls_params = cls._generate_example_payloads_and_steps(data_sources_config)
        if not report_calls_params:
            st.warning(f"视件 '{cls.get_display_name()}' 的 _generate_example_payloads_and_steps 未提供任何数据。")
            return {}

        all_reported_assets: List[Dict[str, Any]] = []
        for params in report_calls_params:
            try:
                payload = params.get("data_payload")
                step = params.get("step")
                if payload is None or step is None:
                    st.warning(f"跳过无效的示例报告参数: {params}")
                    continue
                
                asset_descs = temp_vis_instance.report_data(
                    data_payload=payload,
                    step=int(step),
                    group_id=params.get("group_id"),
                    asset_name=params.get("asset_name")
                )
                all_reported_assets.extend(asset_descs)
            except Exception as e:
                st.error(f"为视件 '{cls.get_display_name()}' 生成示例数据时调用 report_data 失败: {e}")
                st.exception(e) 
        
        if not all_reported_assets:
            st.error(f"视件 '{cls.get_display_name()}' 的 report_data 未能成功返回任何资产描述。")
            return {}

        # --- Logic to construct data_sources_map for the IDE ---
        # This map tells the Viskit instance (when re-instantiated in IDE) where to find the data it just generated.
        
        # Strategy:
        # 1. If all reported assets point to the *same* 'path' (e.g., an aggregating log file),
        #    then the map will have one entry using cls.LOGICAL_DATA_SOURCE_NAME.
        # 2. If reported assets have *different* 'paths' (e.g., one file per step for Treescope),
        #    then the map will have one entry (key cls.LOGICAL_DATA_SOURCE_NAME) whose value
        #    is a "collection" descriptor, containing all individual asset descriptions in an 'items' list.
        #    The 'data_type' of this collection descriptor will be cls.COLLECTION_DATA_TYPE_FOR_IDE.

        unique_asset_paths = {desc['path'] for desc in all_reported_assets if isinstance(desc, dict) and 'path' in desc}
        logical_ds_name_for_map = getattr(cls, 'LOGICAL_DATA_SOURCE_NAME', 'default_data_source')

        if len(unique_asset_paths) == 1: 
            # All reports updated the same underlying asset/file.
            # Use the last asset description as it should be the most up-to-date.
            final_asset_desc = all_reported_assets[-1]
            return {logical_ds_name_for_map: final_asset_desc}
        elif len(unique_asset_paths) > 1: 
            # Multiple distinct assets were reported (e.g., one file per step).
            # The Viskit's load_data method will expect a collection.
            collection_data_type = getattr(cls, 'COLLECTION_DATA_TYPE_FOR_IDE', f"{cls.__name__}_ide_collection_v1")
            
            # Try to infer a display name for the collection
            # This could come from a common group_id or the Viskit's display name
            collection_display_name = f"{cls.get_display_name()} - 示例数据"
            group_id_source = (data_sources_config or {}).get("group_id")
            if not group_id_source and all_reported_assets:
                group_id_source = all_reported_assets[0].get("group_id", ide_instance_id)
            if group_id_source:
                collection_display_name = f"{group_id_source.replace('_',' ').title()} - 示例集合"


            return {
                logical_ds_name_for_map: { 
                    "asset_id": f"collection_for_{ide_instance_id}_{group_id_source}", # Unique ID for the collection itself
                    "data_type": collection_data_type, # The type load_data will check for a collection
                    "display_name": collection_display_name,
                    "items": all_reported_assets, # List of individual asset descriptions
                    "group_id_source": group_id_source # Original group_id if relevant
                }
            }
        else: # No assets with paths were reported
            st.warning(f"视件 '{cls.get_display_name()}' 的 report_data 未返回任何带有效路径的资产描述。")
            return {}


    def _get_config_file_path(self) -> Path:
        return self.private_storage_path / "_ui_config.toml" 

    def save_ui_config(self) -> None:
        config_file = self._get_config_file_path()
        config_to_save = {}
        if PYDANTIC_AVAILABLE and isinstance(self.ui_config, BaseModel):
            config_to_save = self.ui_config.model_dump(mode='json') 
        elif is_dataclass(self.ui_config): 
            from dataclasses import asdict
            config_to_save = asdict(self.ui_config)
        elif isinstance(self.ui_config, dict): 
            config_to_save = self.ui_config
        else:
            st.warning(f"视件 {self.instance_id}: ui_config 不是可识别的可序列化类型，无法保存。")
            return

        try:
            with open(config_file, "wb") as f:
                tomli_w.dump(config_to_save, f)
        except Exception as e:
            st.error(f"保存视件 '{self.instance_id}' UI配置失败: {e}")

    def _load_config_dict_from_file(self) -> Optional[Dict[str, Any]]:
        config_file = self._get_config_file_path()
        if config_file.exists():
            try:
                with open(config_file, "rb") as f:
                    loaded_config_dict = tomli.load(f)
                return loaded_config_dict
            except Exception as e:
                st.error(f"加载视件 '{self.instance_id}' UI配置失败: {e}")
        return None
    
    def render_config_ui(self, config_container) -> bool:
        ConfigModel = self.get_config_model()
        if not (ConfigModel and PYDANTIC_AVAILABLE and STREAMLIT_PYDANTIC_AVAILABLE and issubclass(ConfigModel, BaseModel)):
            config_container.caption("此视件没有可用的Pydantic配置模型或streamlit-pydantic未安装。")
            return False
        
        config_container.warning(
            "Pydantic配置UI渲染的默认实现。子类应覆盖 `render_config_ui`。"
        )
        return False

    @abstractmethod
    def render_report_ui(self, report_container) -> Optional[Dict[str, Any]]:
        pass

