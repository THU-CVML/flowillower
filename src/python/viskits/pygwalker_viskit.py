# flowillower/viskits/pygwalker_viskit.py
import streamlit as st
import pandas as pd
import tomli
import tomli_w
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from math import sin # For example data generation

# Pygwalker and Pydantic imports
try:
    from pygwalker.api.streamlit import StreamlitRenderer
    PYGWALKER_AVAILABLE = True
except ImportError:
    PYGWALKER_AVAILABLE = False
    # Dummy class if pygwalker is not installed
    class StreamlitRenderer:
        def __init__(self, df, spec=None, spec_io_mode='rw', theme_key=None, key=None, **kwargs):
            self.df = df
            st.error("Pygwalker库未安装或无法导入。请运行 'pip install pygwalker'。Pygwalker library not installed or could not be imported. Please run 'pip install pygwalker'.")
        def explorer(self, width: int = None, height: int = None, default_tab:str = "data"): # Added default_tab
            st.warning("Pygwalker 未加载，将显示原始数据表格。Pygwalker not loaded, displaying raw data table instead.")
            st.dataframe(self.df)

from flowillower.viskits.base_viskit import (
    VisKit,
    register_viskit,
    PYDANTIC_AVAILABLE,
    STREAMLIT_PYDANTIC_AVAILABLE
)

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field, field_validator
    from typing_extensions import Literal # For Literal type hint
else:
    class BaseModel: pass
    Literal = str # Fallback for Literal
    # If Pydantic is not available, a fallback dataclass could be used here for ui_config,
    # but the example focuses on Pydantic.
    from dataclasses import dataclass


# Define the UI configuration Pydantic Model for this VisKit
if PYDANTIC_AVAILABLE:
    class PygwalkerUIConfig(BaseModel):
        spec_io_mode: Literal['r', 'w', 'rw'] = Field(default='rw', description="图表规格的读写模式 (I/O mode for chart spec: 'r', 'w', 'rw')")
        theme_key: Literal['vega', 'g2', 'streamlit'] = Field(default='streamlit', description="Pygwalker图表主题 (Theme for Pygwalker chart)")
        kernel_computation: bool = Field(default=True, description="是否启用DuckDB内核计算 (Enable DuckDB kernel computation)")
        explorer_height: int = Field(default=800, ge=400, le=2000, description="Pygwalker探索器的高度（像素）(Height of Pygwalker explorer in pixels)")
        default_tab: Literal['data', 'vis'] = Field(default='data', description="Pygwalker默认打开的标签页 (Default tab to open in Pygwalker)")

        @field_validator('spec_io_mode')
        @classmethod
        def check_spec_io_mode(cls, value):
            if value not in ['r', 'w', 'rw']:
                raise ValueError("spec_io_mode 必须是 'r', 'w', 或 'rw'")
            return value

        @field_validator('theme_key')
        @classmethod
        def check_theme_key(cls, value):
            if value not in ['vega', 'g2', 'streamlit']: # 'streamlit' is a valid theme for pygwalker
                raise ValueError("theme_key 必须是 'vega', 'g2', 或 'streamlit'")
            return value
        
        @field_validator('default_tab')
        @classmethod
        def check_default_tab(cls, value):
            if value not in ['data', 'vis']:
                raise ValueError("default_tab 必须是 'data' 或 'vis'")
            return value

else: # Fallback if Pydantic is not available
    from dataclasses import dataclass
    @dataclass
    class PygwalkerUIConfig: # type: ignore
        spec_io_mode: str = 'rw'
        theme_key: str = 'streamlit'
        kernel_computation: bool = True
        explorer_height: int = 800
        default_tab: str = 'data'


@register_viskit(name="pygwalker_interactive_dashboard")
class PygwalkerDashboardVisKit(VisKit):
    ui_config: PygwalkerUIConfig
    _raw_data_df: Optional[pd.DataFrame] = None

    DATA_FILE_NAME = "pygwalker_scalar_log.toml" # Can share data or have its own
    LOGICAL_DATA_SOURCE_NAME = "pygwalker_metrics_source"
    INTERNAL_DATA_TYPE_NAME = "pygwalker_dashboard_log_v1" # Specific to its own data format if it writes

    def __init__(self,
                 instance_id: str,
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]],
                 specific_ui_config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(instance_id, trial_root_path, data_sources_map, specific_ui_config_dict)
        self._raw_data_df = None
        # self.ui_config (PygwalkerUIConfig) is initialized by the base class

    @classmethod
    def get_config_model(cls) -> Optional[Type[PygwalkerUIConfig]]:
        return PygwalkerUIConfig if PYDANTIC_AVAILABLE else None

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        # This Viskit can consume the same raw scalar data as ScalarMetricsDashboardVisKit,
        # OR it can consume data it logged itself.
        return cls.INTERNAL_DATA_TYPE_NAME in data_type_names or \
               "multi_metric_multi_track_scalars" in data_type_names # For consuming shared data

    @classmethod
    def _generate_example_payloads_and_steps(cls,
                                             data_sources_config: Optional[Dict[str, Any]] = None
                                             ) -> List[Dict[str, Any]]:
        payloads = []
        num_example_steps = (data_sources_config or {}).get("num_steps", 30)
        for step in range(num_example_steps):
            payloads.append({
                "data_payload": {"global_step": step, "track": "alpha", "value_a": sin(step/5) * 10, "value_b": step * 0.5},
                "step": step, "group_id": "alpha_metrics" # group_id becomes track in report_data
            })
            payloads.append({
                "data_payload": {"global_step": step, "track": "beta", "value_a": sin(step/7 + 0.5) * 5, "value_c": 100 - step * 1.5},
                "step": step, "group_id": "beta_metrics"
            })
        return payloads

    def report_data(self,
                    data_payload: Dict[str, Any], # Expects dict where keys can be metrics, or special keys like 'global_step', 'track'
                    step: int, # step from argument is primary
                    group_id: Optional[str] = None, # Used as 'track' if 'track' not in data_payload
                    asset_name: Optional[str] = None,
                    **kwargs) -> List[Dict[str, Any]]:
        """
        Appends data to this Viskit's private TOML log file.
        The data_payload should be a flat dictionary of scalars.
        'global_step' and 'track' can be in data_payload or passed as arguments.
        """
        log_file_path = self.private_storage_path / self.DATA_FILE_NAME
        
        entry_to_log = {}
        # Prioritize step from argument
        entry_to_log["global_step"] = step 
        
        # Handle track: from group_id or from payload
        if "track" in data_payload and isinstance(data_payload["track"], str):
            entry_to_log["track"] = data_payload["track"]
        elif group_id:
            entry_to_log["track"] = group_id
        else:
            entry_to_log["track"] = "default" # Default track if none provided

        valid_scalars = 0
        for key, value in data_payload.items():
            if key in ["global_step", "track"]: # Already handled
                continue
            if isinstance(value, (int, float)):
                entry_to_log[key] = value
                valid_scalars += 1
            else:
                print(f"警告: Pygwalker视件: 指标 '{key}' 的值不是数字 ({value})，将被忽略。")

        if valid_scalars == 0:
            print(f"警告: Pygwalker视件: 在步骤 {step}, 组 '{entry_to_log['track']}' 中没有有效的标量数据可报告。")
            return []

        all_data = {"metrics": []} # Pygwalker expects a list of records
        if log_file_path.exists():
            try:
                with open(log_file_path, "rb") as f:
                    all_data = tomli.load(f)
                    if "metrics" not in all_data or not isinstance(all_data["metrics"], list):
                        all_data["metrics"] = []
            except Exception as e:
                print(f"读取现有Pygwalker日志 {log_file_path} 失败: {e}。将创建新文件。")
                all_data["metrics"] = []
        
        all_data["metrics"].append(entry_to_log)

        try:
            with open(log_file_path, "wb") as f:
                tomli_w.dump(all_data, f)
        except Exception as e:
            print(f"写入Pygwalker日志 {log_file_path} 失败: {e}")
            return []

        relative_log_path = (self.private_storage_path.relative_to(self.trial_root_path) / self.DATA_FILE_NAME).as_posix()
        asset_description = {
            "asset_id": f"{self.instance_id}_pygwalker_log",
            "display_name": f"{self.get_display_name()} - {self.instance_id}",
            "data_type_original": self.INTERNAL_DATA_TYPE_NAME, # This Viskit's own data format
            "path": relative_log_path,
            "group_id": self.instance_id
        }
        return [asset_description]

    def load_data(self) -> None:
        data_asset_info = self._get_data_asset_info(self.LOGICAL_DATA_SOURCE_NAME)
        data_asset_path = None
        if data_asset_info and "path" in data_asset_info:
            data_asset_path = (self.trial_root_path / data_asset_info["path"]).resolve()
        
        if data_asset_path is None or not data_asset_path.exists():
            # Fallback to its own private storage if not found via manifest/data_sources_map
            data_asset_path = self.private_storage_path / self.DATA_FILE_NAME

        if not data_asset_path.exists():
            self._raw_data_df = pd.DataFrame()
            self._all_available_steps = []
            return

        try:
            with open(data_asset_path, "rb") as f:
                data = tomli.load(f)
            metrics_list = data.get("metrics", [])
            self._raw_data_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
            
            if "global_step" in self._raw_data_df.columns:
                self._raw_data_df["global_step"] = pd.to_numeric(self._raw_data_df["global_step"], errors='coerce')
                valid_steps = self._raw_data_df["global_step"].dropna().astype(int).unique()
                self._all_available_steps = sorted(list(valid_steps))
            else:
                self._all_available_steps = []

        except Exception as e:
            st.error(f"视件 {self.instance_id}: 加载Pygwalker数据 '{data_asset_path}' 失败: {e}")
            self._raw_data_df = pd.DataFrame()
            self._all_available_steps = []
            
    def render_config_ui(self, config_container) -> bool:
        """Renders config UI for PygwalkerUIConfig."""
        if not (PYDANTIC_AVAILABLE and STREAMLIT_PYDANTIC_AVAILABLE):
            config_container.caption("Pydantic 或 streamlit-pydantic 未安装，无法渲染配置表单。")
            return False
        
        # 使用 streamlit-pydantic 生成表单
        # Use streamlit-pydantic to generate the form
        # pydantic_form 在提交时返回新的模型实例
        # pydantic_form returns a new model instance on submission
        # 我们需要捕获它并更新 self.ui_config
        # We need to capture it and update self.ui_config
        form_key = f"{self.instance_id}_pyg_config_form"
        
        # 为了让表单显示当前值，我们需要将 self.ui_config 实例传给它
        # To make the form show current values, we need to pass the self.ui_config instance
        
        # Hack: streamlit-pydantic form doesn't directly support instance editing with bool return for change.
        # We'll use a sub-container and st.form for manual control if needed, or rely on pydantic_form's behavior.
        # For now, let's try passing the instance.
        
        # streamlit-pydantic's pydantic_form can take an instance for default values.
        # It returns the *new* instance upon submission.
        with config_container.form(key=form_key + "_manual"): # Use st.form for explicit submission tracking
            # Manually create fields based on self.ui_config Pydantic model
            current_config = self.ui_config
            new_values = {}

            new_values["spec_io_mode"] = st.selectbox(
                "图表规格交互模式", options=['rw', 'r', 'w'],
                index=['rw', 'r', 'w'].index(current_config.spec_io_mode),
                key=f"{form_key}_spec_io"
            )
            new_values["theme_key"] = st.selectbox(
                "图表主题", options=['streamlit', 'vega', 'g2'], # Ensure 'streamlit' is an option
                index=['streamlit', 'vega', 'g2'].index(current_config.theme_key),
                key=f"{form_key}_theme"
            )
            new_values["kernel_computation"] = st.checkbox(
                "启用内核计算", value=current_config.kernel_computation,
                key=f"{form_key}_kernel"
            )
            new_values["explorer_height"] = st.number_input(
                "探索器高度", min_value=300, max_value=2000,
                value=current_config.explorer_height, step=50,
                key=f"{form_key}_height"
            )
            new_values["default_tab"] = st.selectbox(
                "默认标签页", options=['data', 'vis'],
                index=['data', 'vis'].index(current_config.default_tab),
                key=f"{form_key}_default_tab"
            )

            submitted = st.form_submit_button("应用Pygwalker设置")

            if submitted:
                try:
                    updated_model = PygwalkerUIConfig(**new_values)
                    if self.ui_config.model_dump() != updated_model.model_dump():
                        self.ui_config = updated_model
                        self.save_ui_config()
                        return True # Config changed
                except ValidationError as ve:
                    st.error(f"Pygwalker配置验证失败: {ve}")
                except Exception as e_cfg:
                    st.error(f"应用Pygwalker配置时出错: {e_cfg}")
        return False


    def render_report_ui(self, report_container) -> Optional[Dict[str, Any]]:
        """为Pygwalker视件渲染用于手动触发report_data的UI。"""
        report_container.markdown(f"#### 上报标量数据到 `{self.instance_id}` (Pygwalker)")
        with report_container.form(key=f"{self.instance_id}_pyg_report_form"):
            step = st.number_input("全局步骤 (Global Step)", min_value=0, value=self._current_global_step or 0, step=1)
            group_id = st.text_input("组ID / Track名称 (Group ID / Track Name)", value="pyg_track")
            
            st.markdown("**指标 (Metrics):** (名称: 值)")
            if f"{self.instance_id}_pyg_report_metrics" not in st.session_state:
                st.session_state[f"{self.instance_id}_pyg_report_metrics"] = [{"metric_name": "cpu_usage", "value": 75.0}]

            edited_metrics = st.data_editor(
                st.session_state[f"{self.instance_id}_pyg_report_metrics"],
                num_rows="dynamic",
                key=f"{self.instance_id}_pyg_report_metrics_editor",
                column_config={
                    "metric_name": st.column_config.TextColumn("指标名称", required=True),
                    "value": st.column_config.NumberColumn("值", format="%.4f", required=True),
                }
            )
            st.session_state[f"{self.instance_id}_pyg_report_metrics"] = edited_metrics

            submit_button = st.form_submit_button("上报数据 (Report Data)")

            if submit_button:
                data_payload = {}
                valid_payload = True
                for item in edited_metrics:
                    name = item.get("metric_name")
                    value = item.get("value")
                    if name and isinstance(name, str) and name.strip() and isinstance(value, (int, float)):
                        data_payload[name.strip()] = float(value)
                    elif name or value is not None: 
                        st.error(f"无效的指标条目: 名称='{name}', 值='{value}'。")
                        valid_payload = False; break 
                
                if not data_payload and valid_payload: 
                    st.warning("请输入至少一个有效的指标和值。"); return None
                
                if valid_payload and data_payload:
                    return {"data_payload": data_payload, "step": int(step), "group_id": group_id if group_id.strip() else None}
        return None

    def render(self) -> None:
        st.subheader(self.get_display_name())

        with st.expander("Pygwalker 显示设置", expanded=False):
            if self.render_config_ui(st.container()):
                st.rerun()

        if self._raw_data_df is None: self.load_data()
        if self._raw_data_df is None or self._raw_data_df.empty:
            st.info(f"视件 {self.instance_id}: 没有数据可供Pygwalker显示。")
            return
        
        if not PYGWALKER_AVAILABLE: # 如果Pygwalker未加载，则显示错误并回退
            st.error("Pygwalker库未加载，无法渲染交互式仪表盘。将显示原始数据。")
            st.dataframe(self._raw_data_df)
            return

        try:
            pyg_key = f"pygwalker_explorer_{self.instance_id}"
            
            # @st.cache_resource # Pygwalker 0.4+ 推荐缓存渲染器
            # TODO 缓存了，无法看到新数据
            def get_pyg_renderer(_df, _spec, _spec_io_mode, _theme_key, _kernel_comp, _key_suffix):
                # 使用唯一key确保每个实例的缓存独立
                # Use unique key to ensure cache is independent for each instance
                # print(f"Creating/retrieving pygwalker renderer with key: {_key_suffix}")
                return StreamlitRenderer(
                    _df,
                    spec=_spec, # spec通常是JSON字符串或文件路径
                    spec_io_mode=_spec_io_mode,
                    theme_key=_theme_key,
                    kernel_computation=_kernel_comp,
                    key=f"pyg_internal_{_key_suffix}" # 内部key，确保Pygwalker状态独立
                )
            renderer = get_pyg_renderer(
                self._raw_data_df.copy(), # 传递副本以防万一 Pass a copy just in case
               (self.private_storage_path / "pyg_spec.json").as_posix(), # 从配置或默认路径加载spec
                self.ui_config.spec_io_mode,
                self.ui_config.theme_key,
                self.ui_config.kernel_computation,
                self.instance_id # 用于缓存key的后缀 Suffix for cache key
            )
            
            # 不使用st.cache_resource的直接实例化，因为spec路径可能在UI中更改
            # Direct instantiation without st.cache_resource as spec path might change via UI
            # Pygwalker的StreamlitRenderer内部似乎有自己的状态管理，基于传递给explorer的key
            # Pygwalker's StreamlitRenderer seems to have its own state management internally, based on the key passed to explorer
            
            # spec路径应该是组件私有存储的一部分
            # spec path should be part of component's private storage
            # spec_file_path = (self.private_storage_path / f"{self.instance_id}_pyg_config.json").as_posix()

            # renderer = StreamlitRenderer(
            #     self._raw_data_df,
            #     spec=spec_file_path, # Pygwalker将在此路径读写spec文件 Pygwalker will read/write spec file at this path
            #     spec_io_mode=self.ui_config.spec_io_mode,
            #     theme_key=self.ui_config.theme_key,
            #     kernel_computation=self.ui_config.kernel_computation,
            #     # `key` 参数对于 `StreamlitRenderer` 的 `explorer` 方法很重要，以区分不同的Pygwalker实例
            #     # The `key` parameter is important for `StreamlitRenderer`'s `explorer` method
            #     # to differentiate between different Pygwalker instances on the same page.
            #     # 然而，StreamlitRenderer构造函数本身不接受key，它在explorer中隐式使用。
            #     # However, the StreamlitRenderer constructor itself doesn't take a key; it's used implicitly in explorer.
            #     # 我们需要确保Pygwalker UI的唯一性。
            #     # We need to ensure uniqueness for Pygwalker UI.
            #     # Pygwalker 0.4+ 的 StreamlitRenderer 构造函数接受 `key`
            #     # Pygwalker 0.4+ StreamlitRenderer constructor accepts `key`
            #     # key=pyg_key # 传递给构造函数 Pass to constructor
            # )
            renderer.explorer(
                default_tab=self.ui_config.default_tab, 
                # height=self.ui_config.explorer_height
            )

        except NameError as ne: 
            if "StreamlitRenderer" in str(ne):
                 st.error("Pygwalker 渲染器不可用。请检查安装。")
                 st.dataframe(self._raw_data_df) 
            else: raise ne 
        except Exception as e:
            st.error(f"渲染Pygwalker组件时出错: {e}")
            st.exception(e) # 打印完整堆栈跟踪
            st.dataframe(self._raw_data_df) 
