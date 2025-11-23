# ./utils/residual_prompt.py
import json
import os

class ResidualPromptDB:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
        # 统一成绝对路径，避免相对路径对不上
        self.db = {
            os.path.abspath(k): v
            for k, v in self.db.items()
        }

    def get_text_pair(self, img_path: str):
        """
        返回两段字符串：
          txt_sem: "amusement park rides, colorful umbrellas, ..."
          txt_tex: "water surface reflections, sand grain texture, ..."
        如果找不到就返回 ("", "")
        """
        key = os.path.abspath(img_path)
        item = self.db.get(key)
        if item is None:
            return "", ""

        sem_list = item.get("semantic_residual", []) or []
        tex_list = item.get("texture_residual", []) or []

        txt_sem = ", ".join(sem_list)
        txt_tex = ", ".join(tex_list)
        return txt_sem, txt_tex

    def get_full_prompt(self, img_path: str):
        """
        拼一个完整 prompt，可以直接喂 text encoder：
        "semantic details: ... texture details: ..."
        """
        txt_sem, txt_tex = self.get_text_pair(img_path)
        if not txt_sem and not txt_tex:
            return ""

        parts = []
        if txt_sem:
            parts.append("semantic details to restore: " + txt_sem)
        if txt_tex:
            parts.append("texture details to restore: " + txt_tex)
        return " ".join(parts)
