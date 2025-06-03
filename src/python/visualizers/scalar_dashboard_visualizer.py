# src/visualizers/scalar_dashboard_visualizer.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tomli
import tomli_w
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .base_visualizer import VisualizationComponent, register_visualizer


@register_visualizer(name="scalar_metrics_dashboard")
class ScalarMetricsDashboardVisualizer(VisualizationComponent):
    """
    一个用于显示多个标量指标（每个指标可能包含多个track）的仪表盘组件。
    它会为每个指标在选定的global_step显示st.metric摘要，并绘制一个Plotly折线图。
    A dashboard component to display multiple scalar metrics, each potentially with multiple tracks.
    It shows st.metric summaries for the selected global_step and a Plotly line chart for each metric.
    """

    _raw_data: Optional[pd.DataFrame] = None
    _processed_metrics_data: Dict[str, pd.DataFrame] = None

    DEFAULT_CHARTS_PER_ROW = 2

    def __init__(
        self,
        component_instance_id: str,
        trial_root_path: Path,
        data_sources_map: Dict[str, Dict[str, Any]],
        component_specific_config: Dict[str, Any] = None,
    ):
        super().__init__(
            component_instance_id,
            trial_root_path,
            data_sources_map,
            component_specific_config,
        )
        self._raw_data = None
        self._processed_metrics_data = {}
        self.load_component_config()

        self.charts_per_row = self.config.get(
            "charts_per_row", self.DEFAULT_CHARTS_PER_ROW
        )
        self.chart_height = self.config.get("chart_height", 400)

    @classmethod
    def can_handle_data_types(cls, data_type_names: List[str]) -> bool:
        return "multi_metric_multi_track_scalars" in data_type_names

    @classmethod
    def generate_example_data(
        cls,
        example_data_path: Path,  # This path is like .../trial_root/example_assets_for_ide
        data_sources_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        example_data_path.mkdir(parents=True, exist_ok=True)
        metrics_file_name = "example_scalar_metrics.toml"
        # 文件实际写入位置: example_data_path / metrics_file_name
        # Actual file write location: example_data_path / metrics_file_name
        metrics_file_full_path = example_data_path / metrics_file_name

        data_points = []
        for step in range(20):
            data_points.append(
                {
                    "global_step": step,
                    "track": "train",
                    "loss": 1.0 / (step + 1) + 0.1,
                    "accuracy": 0.6 + step * 0.015,
                }
            )
            data_points.append(
                {
                    "global_step": step,
                    "track": "validation",
                    "loss": 1.0 / (step + 1) + 0.2,
                    "accuracy": 0.55 + step * 0.01,
                }
            )
            if step % 5 == 0:
                data_points.append(
                    {
                        "global_step": step,
                        "track": "test",
                        "loss": 1.0 / (step + 1) + 0.3,
                        "accuracy": 0.5 + step * 0.005,
                    }
                )
                data_points.append(
                    {
                        "global_step": step,
                        "track": "system",
                        "learning_rate": 0.001 * (0.9**step),
                    }
                )

        try:
            with open(metrics_file_full_path, "wb") as f:
                tomli_w.dump({"metrics": data_points}, f)
        except Exception as e:
            st.error(f"生成示例数据失败: {e}")
            raise

        # *** 修正点 (FIXED POINT) ***
        # 返回的路径应该是相对于 trial_root_path 的。
        # The returned path should be relative to trial_root_path.
        # example_data_path 的父目录是 trial_root_path。
        # The parent of example_data_path is trial_root_path in the IDE's context.
        # 所以，相对路径是 example_data_path 的名称（即 "example_assets_for_ide"）加上文件名。
        # So, the relative path is the name of example_data_path (i.e., "example_assets_for_ide") plus the filename.
        # path_relative_to_trial_root = example_data_path.name / metrics_file_name
        # 更稳健的方式是，如果 example_data_path 是绝对路径，而我们需要相对于 trial_root_path 的路径，
        # 并且我们知道 trial_root_path 是 example_data_path 的父目录（或更早的祖先）。
        # A more robust way, if example_data_path is absolute, and we need path relative to trial_root_path,
        # and we know trial_root_path is a parent (or earlier ancestor) of example_data_path.
        # 在IDE的上下文中，example_data_path = trial_root_path / "example_assets_for_ide"
        # In the IDE's context, example_data_path = trial_root_path / "example_assets_for_ide"
        # 所以，相对于 trial_root_path 的路径就是 "example_assets_for_ide" / metrics_file_name
        # So, path relative to trial_root_path is "example_assets_for_ide" / metrics_file_name

        path_for_manifest = Path(example_data_path.name) / metrics_file_name

        return {
            "main_metrics_source": {
                "asset_id": "example_metrics_dashboard_data",
                "data_type": "multi_metric_multi_track_scalars",
                "path": str(
                    path_for_manifest
                ),  # 使用修正后的相对路径 Use the corrected relative path
                "display_name": "示例综合指标数据 (Example Comprehensive Metrics)",
            }
        }

    def load_data(self) -> None:
        data_asset_path = self._get_data_asset_path("main_metrics_source")
        if data_asset_path is None or not data_asset_path.exists():
            st.warning(
                f"组件 {self.component_instance_id}: 未找到数据源 'main_metrics_source' 或路径无效: {data_asset_path}"
            )
            self._raw_data = pd.DataFrame()
            self._processed_metrics_data = {}
            return

        try:
            with open(data_asset_path, "rb") as f:
                data = tomli.load(f)

            metrics_list = data.get("metrics", [])
            if not metrics_list:
                self._raw_data = pd.DataFrame()
            else:
                self._raw_data = pd.DataFrame(metrics_list)

            self._process_raw_data()

        except Exception as e:
            st.error(
                f"组件 {self.component_instance_id}: 加载数据 '{data_asset_path}' 失败: {e}"
            )
            self._raw_data = pd.DataFrame()
            self._processed_metrics_data = {}

    def _process_raw_data(self) -> None:
        self._processed_metrics_data = {}
        if self._raw_data is None or self._raw_data.empty:
            return

        if "global_step" not in self._raw_data.columns:
            st.warning(
                f"组件 {self.component_instance_id}: 数据缺少 'global_step' 列。"
            )
            return

        potential_metric_cols = [
            col for col in self._raw_data.columns if col not in ["global_step", "track"]
        ]

        for metric_col_name in potential_metric_cols:
            metric_df_cols = ["global_step", metric_col_name]
            if "track" in self._raw_data.columns:
                metric_df_cols.append("track")

            # 确保所有需要的列都存在于self._raw_data中
            # Ensure all needed columns exist in self._raw_data
            if not all(col in self._raw_data.columns for col in metric_df_cols):
                # st.warning(f"组件 {self.component_instance_id}: 指标 '{metric_col_name}' 的原始数据缺少某些列: {metric_df_cols}")
                continue

            metric_df = self._raw_data[metric_df_cols].copy()

            if "track" not in metric_df.columns:  # 如果原始数据就没有track列
                metric_df["track"] = "default"

            metric_df.rename(columns={metric_col_name: "value"}, inplace=True)
            metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
            metric_df["global_step"] = pd.to_numeric(
                metric_df["global_step"], errors="coerce"
            )
            metric_df.dropna(subset=["value", "global_step"], inplace=True)

            if not metric_df.empty:
                metric_df = metric_df.sort_values(
                    by=["track", "global_step"]
                ).reset_index(drop=True)
                self._processed_metrics_data[metric_col_name] = metric_df

        if (
            self._all_available_steps is None
            and not self._raw_data.empty
            and "global_step" in self._raw_data.columns
        ):
            valid_steps = (
                pd.to_numeric(self._raw_data["global_step"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            self._all_available_steps = sorted(list(valid_steps))

    def _render_metric_summary(
        self, metric_name: str, metric_df: pd.DataFrame, target_step: Optional[int]
    ):
        if (
            target_step is None and self._all_available_steps
        ):  # 如果target_step是None，默认用最后一个step
            target_step = self._all_available_steps[-1]
        elif target_step is None or not self._all_available_steps:
            st.metric(label=f"{metric_name}", value="无可用步骤", delta=None)
            return

        actual_display_step = self._get_closest_available_step(target_step)
        if actual_display_step is None:
            st.metric(label=f"{metric_name}", value="无数据", delta=None)
            return

        all_tracks = sorted(list(metric_df["track"].unique()))
        num_tracks = len(all_tracks)
        if num_tracks == 0:
            return

        cols = st.columns(num_tracks) if num_tracks > 1 else [st.container()]

        for idx, track_name in enumerate(all_tracks):
            with cols[idx if num_tracks > 1 else 0]:
                track_data = metric_df[metric_df["track"] == track_name]
                if track_data.empty:
                    st.metric(
                        label=f"{metric_name} ({track_name})",
                        value="无数据",
                        delta=None,
                    )
                    continue

                current_value = None
                delta_value = None
                step_for_current_value = actual_display_step

                current_step_data = track_data[
                    track_data["global_step"] == step_for_current_value
                ]
                if current_step_data.empty:
                    prev_steps_for_track = track_data[
                        track_data["global_step"] <= step_for_current_value
                    ]["global_step"]
                    if not prev_steps_for_track.empty:
                        step_for_current_value = prev_steps_for_track.max()
                        current_step_data = track_data[
                            track_data["global_step"] == step_for_current_value
                        ]

                if not current_step_data.empty:
                    current_value = current_step_data["value"].iloc[0]
                    prev_steps_for_delta = track_data[
                        track_data["global_step"] < step_for_current_value
                    ]["global_step"]
                    if not prev_steps_for_delta.empty:
                        step_for_prev_value = prev_steps_for_delta.max()
                        prev_value_data = track_data[
                            track_data["global_step"] == step_for_prev_value
                        ]
                        if not prev_value_data.empty:
                            prev_value = prev_value_data["value"].iloc[0]
                            delta_value = current_value - prev_value

                metric_label = f"{metric_name} ({track_name})"
                if (
                    step_for_current_value != target_step and current_value is not None
                ):  # 仅当找到的值的步骤与目标步骤不同时才显示
                    metric_label += f" @S{int(step_for_current_value)}"

                st.metric(
                    label=metric_label,
                    value=f"{current_value:.4f}"
                    if current_value is not None
                    else "无数据",
                    delta=f"{delta_value:.4f}"
                    if delta_value is not None and current_value is not None
                    else None,
                )

    def _render_plotly_chart(
        self, metric_name: str, metric_df: pd.DataFrame, chart_key: str
    ):
        fig = go.Figure()
        all_tracks = sorted(list(metric_df["track"].unique()))
        colors = px.colors.qualitative.Plotly

        for i, track_name in enumerate(all_tracks):
            track_data = metric_df[metric_df["track"] == track_name]
            fig.add_trace(
                go.Scatter(
                    x=track_data["global_step"],
                    y=track_data["value"],
                    mode="lines+markers",
                    name=track_name,
                    line=dict(color=colors[i % len(colors)]),
                    marker=dict(
                        size=6,
                        color=colors[i % len(colors)],
                        line=dict(width=1, color="white"),
                    ),
                    customdata=track_data[["global_step", "value", "track"]],
                    hovertemplate="<b>%{customdata[2]}</b><br>Step: %{customdata[0]}<br>Value: %{customdata[1]:.4f}<extra></extra>",
                )
            )

        current_step_to_highlight = self._get_closest_available_step(
            self._current_global_step
        )
        if current_step_to_highlight is not None:
            fig.add_vline(
                x=current_step_to_highlight,
                line_width=1.5,
                line_dash="solid",
                line_color="firebrick",
                opacity=0.7,
            )

        fig.update_layout(
            xaxis_title="Global Step",
            yaxis_title=metric_name,
            height=self.chart_height,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=len(all_tracks) > 1,
            hovermode="closest",
        )

        event_data = st.plotly_chart(
            fig, use_container_width=True, key=chart_key, on_select="rerun"
        )

        current_selection = st.session_state.get(chart_key, {}).get("selection")
        if current_selection and current_selection.get("points"):
            clicked_point = current_selection["points"][0]
            if "customdata" in clicked_point and isinstance(
                clicked_point["customdata"], list
            ):
                clicked_global_step = int(clicked_point["customdata"][0])
                last_clicked_step_key = f"{chart_key}_last_clicked_step"
                if st.session_state.get(last_clicked_step_key) != clicked_global_step:
                    st.session_state[last_clicked_step_key] = clicked_global_step
                    self._request_global_step_change(clicked_global_step)
        elif current_selection and not current_selection.get("points"):
            st.session_state[f"{chart_key}_last_clicked_step"] = None

    def render_settings_ui(self):
        st.markdown("##### 组件设置 (Component Settings)")
        new_charts_per_row = st.slider(
            "每行图表数 (Charts per row)",
            1,
            4,
            self.charts_per_row,
            key=f"{self.component_instance_id}_charts_per_row",
        )
        if new_charts_per_row != self.charts_per_row:
            self.charts_per_row = new_charts_per_row
            self.config["charts_per_row"] = new_charts_per_row
            self.save_component_config()
            st.rerun()

        new_chart_height = st.number_input(
            "图表高度 (Chart Height)",
            min_value=200,
            max_value=1000,
            step=50,
            value=self.chart_height,
            key=f"{self.component_instance_id}_chart_height",
        )
        if new_chart_height != self.chart_height:
            self.chart_height = new_chart_height
            self.config["chart_height"] = new_chart_height
            self.save_component_config()
            st.rerun()

    def render(self) -> None:
        if self._processed_metrics_data is None or not self._processed_metrics_data:
            if self._raw_data is None:  # 尝试加载一次
                self.load_data()
            # 再次检查
            if self._processed_metrics_data is None or not self._processed_metrics_data:
                st.info(
                    f"组件 {self.component_instance_id}: 没有处理好的指标数据可供显示。请先加载数据或生成示例数据。"
                )
                st.caption(
                    "如果已生成示例数据但仍看到此消息，请检查数据路径和格式是否正确。If example data was generated but you still see this, check data paths and format."
                )
                return

        with st.expander("图表显示设置 (Chart Display Settings)", expanded=False):
            self.render_settings_ui()

        metric_names_to_display = sorted(list(self._processed_metrics_data.keys()))
        if not metric_names_to_display:
            st.caption("没有可显示的指标。No metrics to display.")
            return

        num_metrics = len(metric_names_to_display)
        cols_per_row = self.charts_per_row

        for i in range(0, num_metrics, cols_per_row):
            metric_chunk = metric_names_to_display[i : i + cols_per_row]
            actual_cols_for_this_row = len(metric_chunk)
            chart_cols = st.columns(actual_cols_for_this_row)

            for j, metric_name in enumerate(metric_chunk):
                with chart_cols[j]:
                    metric_df = self._processed_metrics_data[metric_name]

                    with st.container(
                        border=True, height=self.chart_height + 200
                    ):  # 增加容器高度以容纳metric和图表
                        st.subheader(metric_name)
                        self._render_metric_summary(
                            metric_name, metric_df, self._current_global_step
                        )
                        st.markdown("---")
                        chart_key = f"plotly_{self.component_instance_id}_{metric_name}"
                        self._render_plotly_chart(metric_name, metric_df, chart_key)
