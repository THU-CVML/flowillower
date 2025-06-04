# flowillower/viskits/scalar_dashboard_viskit.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tomli
import tomli_w
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from math import sin 

from flowillower.viskits.base_viskit import ( 
    VisKit, 
    register_viskit, 
    PYDANTIC_AVAILABLE,
    STREAMLIT_PYDANTIC_AVAILABLE
)

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field
    if STREAMLIT_PYDANTIC_AVAILABLE:
        import streamlit_pydantic as sp
else: 
    class BaseModel: pass 
    from dataclasses import dataclass


if PYDANTIC_AVAILABLE:
    class ScalarDashboardUIConfig(BaseModel):
        charts_per_row: int = Field(default=2, ge=1, le=4, description="每行显示的图表数量")
        chart_height: int = Field(default=400, ge=200, le=1000, description="每个图表的高度（像素）")
        show_metric_summary: bool = Field(default=True, description="是否显示指标摘要 (st.metric)")
else: 
    @dataclass
    class ScalarDashboardUIConfig: # type: ignore
        charts_per_row: int = 2
        chart_height: int = 400
        show_metric_summary: bool = True


@register_viskit(name="scalar_metrics_dashboard") 
class ScalarMetricsDashboardVisKit(VisKit): 
    ui_config: ScalarDashboardUIConfig 
    _raw_data_df: Optional[pd.DataFrame] = None
    _processed_metrics_data: Dict[str, pd.DataFrame] = None 

    DATA_FILE_NAME = "scalar_metrics_log.toml"
    LOGICAL_DATA_SOURCE_NAME = "logged_metrics_source" 
    INTERNAL_DATA_TYPE_NAME = "scalar_dashboard_log_v1"


    def __init__(self,
                 instance_id: str, 
                 trial_root_path: Path,
                 data_sources_map: Dict[str, Dict[str, Any]],
                 specific_ui_config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(instance_id, trial_root_path, data_sources_map, specific_ui_config_dict)
        self._raw_data_df = None
        self._processed_metrics_data = {}

    @classmethod
    def get_config_model(cls) -> Optional[Type[ScalarDashboardUIConfig]]:
        return ScalarDashboardUIConfig if PYDANTIC_AVAILABLE else None

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        return cls.INTERNAL_DATA_TYPE_NAME in data_type_names

    @classmethod
    def _generate_example_payloads_and_steps(cls,
                                             data_sources_config: Optional[Dict[str, Any]] = None
                                             ) -> List[Dict[str, Any]]:
        """
        实现基类的抽象方法，生成用于多次调用 report_data 的示例参数列表。
        Implements the base class abstract method to generate a list of example parameter dicts
        for multiple calls to report_data.
        """
        payloads = []
        num_example_steps = (data_sources_config or {}).get("num_steps", 25)

        for step in range(num_example_steps):
            payload_train = {"loss": 1.0 / (step + 1) + 0.1 + sin(step/3)*0.05, "accuracy": 0.6 + step * 0.015}
            payloads.append({
                "data_payload": payload_train, "step": step, "group_id": "training" 
            })

            payload_val = {"loss": 1.0 / (step + 1) + 0.2 - sin(step/4)*0.03, "accuracy": 0.55 + step * 0.01}
            payloads.append({
                "data_payload": payload_val, "step": step, "group_id": "validation"
            })

            if step % 5 == 0:
                payload_lr = {"learning_rate": 0.001 * (0.9**(step/5))}
                payloads.append({
                    "data_payload": payload_lr, "step": step, "group_id": "optimizer"
                })
        return payloads

    # generate_example_data 方法现在由基类 VisKit 提供通用实现
    # The generate_example_data method is now provided by the base VisKit class with a generic implementation

    def report_data(self,
                    data_payload: Dict[str, float], 
                    step: int,
                    group_id: Optional[str] = None, 
                    asset_name: Optional[str] = None, 
                    **kwargs) -> List[Dict[str, Any]]:
        if not isinstance(data_payload, dict):
            print(f"错误: {self.__class__.__name__}.report_data 期望 data_payload 是字典，但收到了 {type(data_payload)}")
            return []

        log_file_path = self.private_storage_path / self.DATA_FILE_NAME
        
        new_entries = []
        current_entry_base = {"global_step": step}
        if group_id: 
            current_entry_base["track"] = group_id 
        
        valid_payload_items = 0
        for key, value in data_payload.items():
            if isinstance(value, (int, float)):
                current_entry_base[key] = value
                valid_payload_items +=1
            else:
                print(f"警告: 指标 '{key}' (组 '{group_id}', 步骤 {step}) 的值不是数字 ({value})，将被忽略。")
        
        if valid_payload_items > 0:
            new_entries.append(current_entry_base)
        else:
            print(f"警告: 在步骤 {step}，组 '{group_id}' 中没有有效的标量数据可报告。")
            return []

        all_metrics_data = {"metrics": []}
        if log_file_path.exists():
            try:
                with open(log_file_path, "rb") as f:
                    all_metrics_data = tomli.load(f)
                    if "metrics" not in all_metrics_data or not isinstance(all_metrics_data["metrics"], list):
                        all_metrics_data["metrics"] = [] 
            except Exception as e:
                print(f"读取现有标量日志 {log_file_path} 失败: {e}。将创建新文件。")
                all_metrics_data["metrics"] = []
        
        all_metrics_data["metrics"].extend(new_entries)

        try:
            with open(log_file_path, "wb") as f:
                tomli_w.dump(all_metrics_data, f)
        except Exception as e:
            print(f"写入标量日志 {log_file_path} 失败: {e}")
            return [] 

        relative_log_path = (self.private_storage_path.relative_to(self.trial_root_path) / self.DATA_FILE_NAME).as_posix()

        asset_description = {
            "asset_id": f"{self.instance_id}_scalar_log", 
            "display_name": f"{self.get_display_name()} - {self.instance_id}", 
            "data_type_original": self.INTERNAL_DATA_TYPE_NAME, 
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
            data_asset_path = self.private_storage_path / self.DATA_FILE_NAME

        if not data_asset_path.exists():
            self._raw_data_df = pd.DataFrame() 
            self._processed_metrics_data = {}
            self._all_available_steps = [] 
            return

        try:
            with open(data_asset_path, "rb") as f:
                data = tomli.load(f)
            metrics_list = data.get("metrics", [])
            self._raw_data_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
            self._process_raw_data()
        except Exception as e:
            st.error(f"视件 {self.instance_id}: 加载数据 '{data_asset_path}' 失败: {e}")
            self._raw_data_df = pd.DataFrame()
            self._processed_metrics_data = {}
            self._all_available_steps = []


    def _process_raw_data(self) -> None: 
        self._processed_metrics_data = {}
        if self._raw_data_df is None or self._raw_data_df.empty:
            self._all_available_steps = [] 
            return

        if "global_step" not in self._raw_data_df.columns:
            self._all_available_steps = []
            st.warning(f"视件 {self.instance_id}: 数据缺少 'global_step' 列。")
            return
        
        self._raw_data_df["global_step"] = pd.to_numeric(self._raw_data_df["global_step"], errors='coerce')
        self._raw_data_df.dropna(subset=["global_step"], inplace=True) 
        if self._raw_data_df.empty: 
            self._all_available_steps = []
            return
        self._raw_data_df["global_step"] = self._raw_data_df["global_step"].astype(int)

        potential_metric_cols = [
            col for col in self._raw_data_df.columns if col not in ["global_step", "track"]
        ]

        for metric_col_name in potential_metric_cols:
            metric_df_cols = ["global_step", metric_col_name]
            if "track" in self._raw_data_df.columns: 
                 metric_df_cols.append("track")
            
            if not all(col in self._raw_data_df.columns for col in metric_df_cols):
                continue

            metric_df = self._raw_data_df[metric_df_cols].copy()
            
            if "track" not in metric_df.columns: 
                metric_df["track"] = "default" 
            
            metric_df.rename(columns={metric_col_name: "value"}, inplace=True)
            metric_df["value"] = pd.to_numeric(metric_df["value"], errors='coerce')
            metric_df.dropna(subset=["value", "global_step"], inplace=True) 
            
            if not metric_df.empty:
                metric_df = metric_df.sort_values(by=["track", "global_step"]).reset_index(drop=True)
                self._processed_metrics_data[metric_col_name] = metric_df
        
        if not self._raw_data_df.empty and "global_step" in self._raw_data_df.columns:
            valid_steps = self._raw_data_df["global_step"].unique() 
            self._all_available_steps = sorted(list(valid_steps))
        else:
            self._all_available_steps = []


    def _render_metric_summary(self, metric_name: str, metric_df: pd.DataFrame, target_step: Optional[int]):
        # (此方法的逻辑与之前版本相同，依赖 self._all_available_steps)
        if target_step is None and self._all_available_steps: 
            target_step = self._all_available_steps[-1] 
        elif target_step is None or not self._all_available_steps:
            st.metric(label=f"{metric_name}", value="无可用步骤", delta=None)
            return

        actual_display_step = self._get_closest_available_step(target_step)
        if actual_display_step is None: 
             st.metric(label=f"{metric_name}", value="无数据点", delta=None) 
             return

        all_tracks = sorted(list(metric_df["track"].unique()))
        num_tracks = len(all_tracks)
        if num_tracks == 0: return 

        cols = st.columns(num_tracks) if num_tracks > 1 else [st.container()] 

        for idx, track_name in enumerate(all_tracks):
            with cols[idx if num_tracks > 1 else 0]:
                track_data = metric_df[metric_df["track"] == track_name]
                if track_data.empty:
                    st.metric(label=f"{metric_name} ({track_name})", value="无数据", delta=None)
                    continue
                current_value: Optional[float] = None
                delta_value: Optional[float] = None
                step_for_current_value: Optional[int] = actual_display_step 
                current_step_data = track_data[track_data["global_step"] == step_for_current_value]
                if current_step_data.empty: 
                    prev_steps_for_track = track_data[track_data["global_step"] <= step_for_current_value]["global_step"]
                    if not prev_steps_for_track.empty:
                        step_for_current_value = prev_steps_for_track.max() 
                        current_step_data = track_data[track_data["global_step"] == step_for_current_value]
                if not current_step_data.empty:
                    current_value = float(current_step_data["value"].iloc[0])
                    prev_steps_for_delta = track_data[track_data["global_step"] < step_for_current_value]["global_step"]
                    if not prev_steps_for_delta.empty:
                        step_for_prev_value = prev_steps_for_delta.max()
                        prev_value_data = track_data[track_data["global_step"] == step_for_prev_value]
                        if not prev_value_data.empty:
                            prev_value = float(prev_value_data["value"].iloc[0])
                            delta_value = current_value - prev_value 
                metric_label = f"{metric_name} ({track_name})"
                if step_for_current_value is not None and step_for_current_value != target_step and current_value is not None:
                    metric_label += f" @S{int(step_for_current_value)}"
                st.metric(
                    label=metric_label,
                    value=f"{current_value:.4f}" if current_value is not None else "无数据",
                    delta=f"{delta_value:.4f}" if delta_value is not None and current_value is not None else None,
                )

    def _render_plotly_chart(self, metric_name: str, metric_df: pd.DataFrame, chart_key: str): 
        # (此方法的逻辑与之前版本相同，依赖 self.ui_config.chart_height)
        fig = go.Figure()
        all_tracks = sorted(list(metric_df["track"].unique()))
        colors = px.colors.qualitative.Plotly 
        for i, track_name in enumerate(all_tracks):
            track_data = metric_df[metric_df["track"] == track_name]
            fig.add_trace(go.Scatter(
                x=track_data["global_step"], y=track_data["value"], mode="lines+markers", name=track_name,
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=6, color=colors[i % len(colors)], line=dict(width=1, color="white")),
                customdata=track_data[["global_step", "value", "track"]], 
                hovertemplate="<b>%{customdata[2]}</b><br>Step: %{customdata[0]}<br>Value: %{customdata[1]:.4f}<extra></extra>"
            ))
        current_step_to_highlight = self._get_closest_available_step(self._current_global_step)
        if current_step_to_highlight is not None:
            fig.add_vline(x=current_step_to_highlight, line_width=1.5, line_dash="solid", line_color="firebrick", opacity=0.7)
        
        fig.update_layout(
            xaxis_title="Global Step", yaxis_title=metric_name, height=self.ui_config.chart_height,
            margin=dict(l=10, r=10, t=30, b=10), showlegend=len(all_tracks) > 1, hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key, on_select="rerun")
        current_selection = st.session_state.get(chart_key, {}).get("selection")
        if current_selection and current_selection.get("points"):
            clicked_point = current_selection["points"][0]
            if "customdata" in clicked_point and isinstance(clicked_point["customdata"], list) and len(clicked_point["customdata"]) > 0:
                try:
                    clicked_global_step = int(clicked_point["customdata"][0])
                    last_clicked_step_key = f"{chart_key}_last_clicked_step" 
                    if st.session_state.get(last_clicked_step_key) != clicked_global_step:
                        st.session_state[last_clicked_step_key] = clicked_global_step
                        self._request_global_step_change(clicked_global_step)
                except (ValueError, TypeError): pass 
        elif current_selection and not current_selection.get("points"): 
            st.session_state[f"{chart_key}_last_clicked_step"] = None

    def render_config_ui(self, config_container) -> bool:
        if not (PYDANTIC_AVAILABLE): # streamlit-pydantic not strictly needed for manual UI
            config_container.caption("Pydantic 未安装，无法渲染配置表单。")
            return False
        
        ConfigModel = self.get_config_model()
        if not ConfigModel: 
            config_container.caption("此视件没有可配置的UI模型。")
            return False

        changed = False
        config_container.markdown("##### 视件显示设置 (VisKit Display Settings)")

        with config_container.form(key=f"{self.instance_id}_scalar_cfg_form_manual"):
            # 从 self.ui_config (已经是Pydantic模型实例) 获取当前值
            # Get current values from self.ui_config (already a Pydantic model instance)
            current_charts_per_row = self.ui_config.charts_per_row
            current_chart_height = self.ui_config.chart_height
            current_show_summary = self.ui_config.show_metric_summary
            
            new_charts_per_row = st.slider(
                "每行图表数", 1, 4, current_charts_per_row,
                key=f"{self.instance_id}_cfg_charts_per_row_manual"
            )
            new_chart_height = st.number_input(
                "图表高度", min_value=200, max_value=1000, step=50,
                value=current_chart_height,
                key=f"{self.instance_id}_cfg_chart_height_manual"
            )
            new_show_summary = st.checkbox(
                "显示指标摘要", value=current_show_summary,
                key=f"{self.instance_id}_cfg_show_summary_manual"
            )
            
            submitted = st.form_submit_button("应用设置 (Apply Settings)")
            if submitted:
                # 检查是否有任何值发生变化
                # Check if any value has changed
                if new_charts_per_row != current_charts_per_row or \
                   new_chart_height != current_chart_height or \
                   new_show_summary != current_show_summary:
                    try:
                        # 使用新值更新Pydantic模型实例
                        # Update Pydantic model instance with new values
                        self.ui_config.charts_per_row = new_charts_per_row
                        self.ui_config.chart_height = new_chart_height
                        self.ui_config.show_metric_summary = new_show_summary
                        self.save_ui_config()
                        changed = True
                    except ValidationError as ve: # Pydantic模型字段赋值时可能会进行验证
                        st.error(f"配置验证失败: {ve}")
                    except Exception as e_cfg:
                        st.error(f"应用配置时出错: {e_cfg}")
        return changed


    def render_report_ui(self, report_container) -> Optional[Dict[str, Any]]:
        report_container.markdown(f"#### 上报标量数据到 `{self.instance_id}`")
        with report_container.form(key=f"{self.instance_id}_scalar_report_form"):
            step = st.number_input("全局步骤 (Global Step)", min_value=0, value=self._current_global_step or 0, step=1)
            group_id = st.text_input("组ID / Track名称 (Group ID / Track Name)", value="default_track")
            
            st.markdown("**指标 (Metrics):** (名称: 值)")
            # 使用唯一的key，基于实例ID和用途
            # Use a unique key based on instance_id and purpose
            session_key_for_metrics = f"{self.instance_id}_ide_report_metrics"
            if session_key_for_metrics not in st.session_state:
                st.session_state[session_key_for_metrics] = [{"metric_name": "loss", "value": 0.0}]

            edited_metrics = st.data_editor(
                st.session_state[session_key_for_metrics],
                num_rows="dynamic",
                key=f"{self.instance_id}_report_metrics_editor", # 确保key唯一
                column_config={
                    "metric_name": st.column_config.TextColumn("指标名称", required=True),
                    "value": st.column_config.NumberColumn("值", format="%.4f", required=True),
                }
            )
            st.session_state[session_key_for_metrics] = edited_metrics

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
        if self._processed_metrics_data is None or not self._processed_metrics_data:
            if self._raw_data_df is None: 
                self.load_data()
            if self._processed_metrics_data is None or not self._processed_metrics_data: 
                st.info(f"视件 {self.instance_id}: 没有处理好的指标数据可供显示。")
                return
        
        with st.expander("视件显示设置", expanded=False): # Changed title
            if self.render_config_ui(st.container()): 
                st.rerun() 

        metric_names_to_display = sorted(list(self._processed_metrics_data.keys()))
        if not metric_names_to_display:
            st.caption("没有可显示的指标。")
            return

        num_metrics = len(metric_names_to_display)
        cols_per_row = self.ui_config.charts_per_row

        for i in range(0, num_metrics, cols_per_row):
            metric_chunk = metric_names_to_display[i : i + cols_per_row]
            actual_cols_for_this_row = len(metric_chunk)
            chart_cols = st.columns(actual_cols_for_this_row)
            
            for j, metric_name in enumerate(metric_chunk):
                with chart_cols[j]:
                    metric_df = self._processed_metrics_data[metric_name]
                    container_height = self.ui_config.chart_height 
                    if self.ui_config.show_metric_summary: 
                        container_height += 150 
                        if len(metric_df["track"].unique()) > 2 : 
                            container_height += 50 * (len(metric_df["track"].unique()) -2)

                    with st.container(border=True, height=container_height): 
                        st.subheader(metric_name)
                        if self.ui_config.show_metric_summary:
                            self._render_metric_summary(metric_name, metric_df, self._current_global_step)
                            st.markdown("---") 
                        chart_key = f"plotly_{self.instance_id}_{metric_name}" 
                        self._render_plotly_chart(metric_name, metric_df, chart_key)
