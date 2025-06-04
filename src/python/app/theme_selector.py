import streamlit as st
import toml
from pathlib import Path
import os
import time


class ThemeSelector:
    def __init__(
        self, themes_dir=".streamlit/themes", config_path=".streamlit/config.toml"
    ):
        self.themes_dir = Path(themes_dir)
        self.config_path = Path(config_path)
        self.themes = {}
        self.load_themes()

    def load_themes(self):
        """加载所有主题文件"""
        self.themes = {}
        if not self.themes_dir.exists():
            return

        for theme_file in self.themes_dir.glob("*.toml"):
            try:
                theme_data = toml.load(theme_file)

                # 从根级别获取theme_name和theme_poem
                theme_name = theme_data.get("theme_name", theme_file.stem)
                theme_poem = theme_data.get("theme_poem", "")
                theme_config = theme_data.get("theme", {})

                self.themes[theme_name] = {
                    "file": theme_file,
                    "name": theme_name,
                    "poem": theme_poem,
                    "config": theme_config,
                }
            except Exception as e:
                st.warning(f"读取主题文件 {theme_file} 失败: {e}")

    def get_current_theme(self):
        """获取当前主题名称"""
        if not self.config_path.exists():
            return None

        try:
            # config = toml.load(self.config_path)
            # # 从根级别读取theme_name
            # current_theme_name = config.get("theme") or {}
            # current_theme_name = current_theme_name.get("theme_name")
            # return current_theme_name
            theme_toml = self.config_path.parent / "theme.toml"
            theme = toml.load(theme_toml)
            return theme.get("theme_name")

        except Exception:
            return None

    def apply_theme(self, theme_name):
        """应用选定的主题"""
        if theme_name not in self.themes:
            st.error(f"主题 '{theme_name}' 不存在")
            return False

        try:
            # 确保配置目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # 读取现有配置或创建新配置
            config = {}
            if self.config_path.exists():
                try:
                    config = toml.load(self.config_path)
                except Exception:
                    config = {}

            # 添加根级别的theme_name和theme_poem
            # config["theme_name"] = self.themes[theme_name]["name"]
            # config["theme_poem"] = self.themes[theme_name]["poem"]

            # 更新主题配置
            theme_config = self.themes[theme_name]["config"].copy()
            # theme_config["theme_name"] = self.themes[theme_name]["name"]
            # theme_config["theme_poem"] = self.themes[theme_name]["poem"]
            config["theme"] |= theme_config

            # 写入配置文件
            with open(self.config_path, "w", encoding="utf-8") as f:
                toml.dump(config, f)

            theme_toml = self.config_path.parent / "theme.toml"
            with open(theme_toml, "w", encoding="utf-8") as f:
                toml.dump(
                    dict(
                        theme_name=self.themes[theme_name]["name"],
                        theme_poem=self.themes[theme_name]["poem"],
                    ),
                    f,
                )

            return True

        except Exception as e:
            st.error(f"应用主题失败: {e}")
            return False

    def render_theme_selector(self):
        """渲染主题选择器UI"""
        if not self.themes:
            st.warning("未找到可用主题")
            return

        theme_names = list(self.themes.keys())
        current_theme = self.get_current_theme()

        # 确定当前选中的索引
        current_index = 0
        if current_theme and current_theme in theme_names:
            current_index = theme_names.index(current_theme)

        # 主题选择下拉菜单
        selected_theme = st.selectbox(
            "选择主题",
            options=theme_names,
            index=current_index,
            format_func=lambda x: self.themes[x]["name"],
            key="theme_selector_widget",
            label_visibility="collapsed",
        )

        # 如果选择了新主题
        if selected_theme != current_theme:
            if self.apply_theme(selected_theme):
                # 显示主题诗句
                theme_poem = self.themes[selected_theme]["poem"]
                if theme_poem:
                    st.toast(f"✨ {theme_poem}", icon="🎨")
                else:
                    st.toast(f"已切换到主题: {selected_theme}", icon="🎨")

                time.sleep(3)
                # 延迟重新运行以应用主题
                st.rerun()

        return selected_theme


# 全局主题选择器实例
_theme_selector = None


def get_theme_selector():
    """获取全局主题选择器实例"""
    global _theme_selector
    if _theme_selector is None:
        _theme_selector = ThemeSelector()
    return _theme_selector


def render_theme_selector():
    """便捷函数：渲染主题选择器"""
    return get_theme_selector().render_theme_selector()
