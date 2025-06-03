# src/visualizers/base_visualizer.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type, Any, Optional, Callable, List
import streamlit as st

# --- Component Registry ---
VISUALIZER_REGISTRY: Dict[str, Type["VisualizationComponent"]] = {}


def register_visualizer(name: str):
    """
    一个装饰器，用于将可视化组件类注册到全局注册表。
    A decorator to register a visualization component class to the global registry.
    """

    def decorator(cls: Type["VisualizationComponent"]):
        if name in VISUALIZER_REGISTRY:
            # 在调试或开发模式下可能是警告，生产模式下可能是错误
            # In debug/dev mode this might be a warning, in production an error
            print(
                f"警告: 可视化组件 '{name}' 已被注册，将被覆盖。Visualizer '{name}' already registered. Will be overridden."
            )
        VISUALIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_visualizer_class(type_name: str) -> Optional[Type["VisualizationComponent"]]:
    """
    从注册表中获取组件类。
    Gets a component class from the registry.
    """
    return VISUALIZER_REGISTRY.get(type_name)


# --- Abstract Base Class for Visualization Components ---
class VisualizationComponent(ABC):
    """
    可视化组件的抽象基类。
    Abstract base class for all visualization components.
    """

    def __init__(
        self,
        component_instance_id: str,
        trial_root_path: Path,
        # data_asset_info: Dict[str, Any], # 描述此组件主要关联的数据资产信息 (来自清单)
        # Describes the data asset this component is primarily associated with (from manifest)
        # ^ 将被更通用的 data_sources_map 替代
        data_sources_map: Dict[
            str, Dict[str, Any]
        ],  # Key: 逻辑数据源名称, Value: 数据资产信息字典
        # Key: logical data source name, Value: data asset info dict
        # e.g., {"main_scalar_data": asset_info_for_loss, "reference_images": asset_info_for_images}
        component_specific_config: Dict[str, Any] = None,
    ):
        """
        初始化可视化组件。
        Initializes the visualization component.

        Args:
            component_instance_id (str): 此组件在仪表盘上的唯一实例ID。
                                         Unique instance ID for this component on the dashboard.
            trial_root_path (Path): 此组件所属的Trial的根目录路径。
                                    Root directory path of the Trial this component belongs to.
            data_sources_map (Dict[str, Dict[str, Any]]):
                一个字典，映射逻辑数据源名称到具体的数据资产信息。
                数据资产信息字典通常包含 'asset_id', 'display_name', 'data_type', 'path',
                以及其他从 _trial_manifest.toml 中解析得到的元数据。
                A dictionary mapping logical data source names to specific data asset information.
                The data asset info dict typically contains 'asset_id', 'display_name', 'data_type', 'path',
                and other metadata parsed from _trial_manifest.toml.
            component_specific_config (Dict[str, Any], optional):
                特定于此组件实例的配置 (例如，图表标题、颜色等)。
                Configuration specific to this component instance (e.g., chart title, color, etc.).
        """
        self.component_instance_id = component_instance_id
        self.trial_root_path = trial_root_path
        self.data_sources_map = data_sources_map
        self.config = (
            component_specific_config if component_specific_config is not None else {}
        )

        # 组件自身持久化配置或小型数据的路径
        # Path for the component to persist its own configuration or small data
        self.component_private_storage_path = (
            self.trial_root_path / "visualizers_data" / self.component_instance_id
        )
        self.component_private_storage_path.mkdir(parents=True, exist_ok=True)

        self._current_global_step: Optional[int] = None
        self._on_global_step_change_request: Optional[Callable[[int], None]] = None
        self._all_available_steps: Optional[List[int]] = (
            None  # 由主应用或数据加载器填充
        )

    def _get_data_asset_info(
        self, logical_name: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        辅助方法，获取指定逻辑名称的数据资产信息。
        Helper method to get data asset info for a given logical name.
        如果组件只处理一个主要数据源，可以使用 "default" 或在初始化时指定。
        If a component handles one primary data source, "default" or a specific name can be used.
        """
        return self.data_sources_map.get(logical_name)

    def _get_data_asset_path(self, logical_name: str = "default") -> Optional[Path]:
        """获取指定逻辑数据源的绝对路径。Gets the absolute path for a given logical data source."""
        asset_info = self._get_data_asset_info(logical_name)
        if asset_info and "path" in asset_info:
            # 路径在清单中是相对于trial_root_path的
            # Path in manifest is relative to trial_root_path
            return (self.trial_root_path / asset_info["path"]).resolve()
        return None

    def configure_global_step_interaction(
        self,
        current_step: Optional[int],
        all_available_steps: Optional[List[int]],
        on_step_change_request_callback: Optional[Callable[[int], None]],
    ):
        """
        由主应用调用，以配置与全局步骤相关的交互。
        Called by the main application to configure global step related interactions.

        Args:
            current_step (Optional[int]): 当前选中的全局步骤。
                                          The currently selected global step.
            all_available_steps (Optional[List[int]]): 此Trial中所有可用的全局步骤列表 (已排序)。
                                                       A sorted list of all available global steps in this Trial.
            on_step_change_request_callback (Optional[Callable[[int], None]]):
                当组件希望更改全局步骤时调用的回调函数。
                Callback function to be called when the component wishes to change the global step.
        """
        self._current_global_step = current_step
        self._all_available_steps = (
            sorted(list(set(all_available_steps))) if all_available_steps else []
        )
        self._on_global_step_change_request = on_step_change_request_callback

    def _request_global_step_change(self, new_step: int) -> None:
        """
        组件内部调用此方法来请求更改全局共享的global_step。
        Component calls this internally to request a change to the shared global_step.
        """
        if self._on_global_step_change_request:
            if self._all_available_steps and new_step not in self._all_available_steps:
                # 如果请求的步骤无效，可以选择寻找最近的有效步骤或忽略
                # If requested step is invalid, can choose to find nearest valid step or ignore
                # For now, let's assume the interaction (e.g., chart click) provides a valid step from its data
                print(
                    f"警告: 组件 {self.component_instance_id} 请求了一个无效的全局步骤 {new_step}。"
                )
                # Potentially find closest:
                # if self._all_available_steps:
                #     new_step = min(self._all_available_steps, key=lambda x: abs(x - new_step))

            self._on_global_step_change_request(new_step)
        else:
            st.warning(
                f"组件 {self.component_instance_id}: 尝试更改全局步骤，但未设置回调。"
            )

    def _get_closest_available_step(self, target_step: Optional[int]) -> Optional[int]:
        """
        如果目标步骤无效或数据在该步骤不可用，则查找最近的可用步骤。
        Finds the closest available step if the target step is invalid or data isn't available at that step.
        组件的子类可以覆盖此逻辑以适应其特定的数据可用性。
        Subclasses can override this logic for their specific data availability.
        """
        if target_step is None:
            return None
        if not self._all_available_steps:
            return None
        if target_step in self._all_available_steps:
            return target_step

        # 寻找最接近的步骤 (简单的实现)
        # Find the closest step (simple implementation)
        try:
            closest = min(self._all_available_steps, key=lambda x: abs(x - target_step))
            return closest
        except ValueError:  # _all_available_steps为空
            return None

    @abstractmethod
    def load_data(self) -> None:
        """
        加载此组件渲染所需的数据。
        Load data required by this component for rendering.
        实现者应使用 self._get_data_asset_path() 来获取数据文件路径。
        Implementers should use self._get_data_asset_path() to get data file paths.
        数据加载后通常存储在实例变量中。
        Loaded data is typically stored in instance variables.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        将组件渲染为Streamlit UI元素。
        Renders the component as Streamlit UI elements.
        应使用 self._current_global_step (可能通过 self._get_closest_available_step 调整) 来显示对应步骤的数据。
        Should use self._current_global_step (possibly adjusted by self._get_closest_available_step)
        to display data for the corresponding step.
        任何可以触发全局步骤更改的交互都应调用 self._request_global_step_change()。
        Any interaction that can trigger a global step change should call self._request_global_step_change().
        """
        pass

    @classmethod
    @abstractmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        """
        类方法：判断此组件类型是否能处理清单中声明的一个或多个数据类型。
        Class method: Determines if this component type can handle one or more data types
        declared in the manifest.

        Args:
            data_type_names (List[str]): 从数据资产清单中获取的数据类型名称列表。
                                         A list of data type names from a data asset manifest.
                                         (通常，主应用会为每个数据资产调用此方法，列表只包含一个元素)
                                         (Usually, the main app calls this for each data asset, so the list has one element)

        Returns:
            bool: True 如果此组件可以处理至少一种给定的数据类型。
                  True if this component can handle at least one of the given data types.
        """
        pass

    @classmethod
    def get_display_name(cls) -> str:
        """
        类方法：返回此组件类型的用户友好显示名称。
        Class method: Returns a user-friendly display name for this component type.
        """
        # 简单的实现：将类名从CamelCase转换为带空格的标题
        # Simple implementation: Convert CamelCase class name to space-separated title
        name = cls.__name__
        if name.endswith("Visualizer"):
            name = name[: -len("Visualizer")]
        s1 = VISUALIZER_REGISTRY.get(
            name, name
        )  # Fallback to class name if not in registry (should not happen with decorator)
        # Add spaces before capital letters (simple version)
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", " ", s1)

    def save_component_config(self) -> None:
        """
        将当前组件的特定配置 (self.config) 保存到其私有存储路径。
        Saves the current component-specific configuration (self.config) to its private storage path.
        """
        config_file = self.component_private_storage_path / "_component_config.toml"
        try:
            import tomli_w  # 确保已安装 Ensure tomli_w is installed

            with open(config_file, "wb") as f:
                tomli_w.dump(self.config, f)
            # st.toast(f"组件 '{self.component_instance_id}' 配置已保存。")
        except Exception as e:
            st.error(f"保存组件 '{self.component_instance_id}' 配置失败: {e}")

    def load_component_config(self) -> None:
        """
        从其私有存储路径加载组件的特定配置，并更新 self.config。
        Loads the component-specific configuration from its private storage path and updates self.config.
        """
        config_file = self.component_private_storage_path / "_component_config.toml"
        if config_file.exists():
            try:
                import tomli  # 确保已安装 Ensure tomli is installed

                with open(config_file, "rb") as f:
                    loaded_config = tomli.load(f)
                self.config.update(loaded_config)  # 合并加载的配置 Merge loaded config
                # st.toast(f"组件 '{self.component_instance_id}' 配置已加载。")
            except Exception as e:
                st.error(f"加载组件 '{self.component_instance_id}' 配置失败: {e}")
