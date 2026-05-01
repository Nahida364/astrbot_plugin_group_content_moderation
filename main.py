import asyncio
from typing import List

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger, AstrBotConfig
from astrbot.core.agent.tool import ToolSet, FunctionTool, ToolExecResult  # 修正导入
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import Field
from pydantic.dataclasses import dataclass


# ---------- 定义审核工具 ----------
@dataclass
class ReportViolationsTool(FunctionTool[AstrAgentContext]):
    name: str = "report_violations"
    description: str = (
        "上报违规消息的编号列表。"
        "违规类型包括：违法信息、色情内容、辱骂/人身攻击。"
        "参数 violation_ids 是一个整数数组，对应消息列表中的编号（从1开始）。"
        "若没有发现违规消息，请调用此工具并传入空数组 []。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "violation_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "违规消息的编号列表。例如 [1, 4, 5]；无违规则为 []"
                }
            },
            "required": ["violation_ids"]
        }
    )

    async def call(
        self,
        context: AstrAgentContext,  # 类型标注
        violation_ids: List[int]
    ) -> ToolExecResult:
        id_list = violation_ids if violation_ids else []
        return ToolExecResult(
            result=f"记录违规消息编号：{id_list if id_list else '无违规'}",
            success=True
        )


# ---------- 插件主类 ----------
class AutoModeration(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._first_loaded = True

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        if not self._first_loaded:
            return
        self._first_loaded = False

        if self.config.get("auto_add_groups", False):
            await self._auto_add_admin_groups()

        asyncio.create_task(self._scheduled_moderation())

    async def _auto_add_admin_groups(self):
        platform = self.context.get_platform(filter.PlatformAdapterType.AIOCQHTTP)
        if not platform:
            logger.warning("aiocqhttp 平台未加载，无法自动添加群聊")
            return

        client = platform.get_client()
        try:
            login_info = await client.api.call_action("get_login_info")
            self_id = login_info["user_id"]
        except Exception as e:
            logger.error(f"获取机器人自身信息失败: {e}")
            return

        try:
            group_list = await client.api.call_action("get_group_list")
        except Exception as e:
            logger.error(f"获取群列表失败: {e}")
            return

        group_ids: list = self.config.get("group_ids", [])
        added = 0
        for group in group_list:
            gid = group["group_id"]
            try:
                member_info = await client.api.call_action(
                    "get_group_member_info",
                    group_id=gid,
                    user_id=self_id,
                    no_cache=True,
                )
                role = member_info.get("role", "")
                if role in ("owner", "admin"):
                    if gid not in group_ids:
                        group_ids.append(gid)
                        added += 1
            except Exception:
                continue

        if added > 0:
            self.config["group_ids"] = group_ids
            self.config.save_config()
            logger.info(f"自动添加了 {added} 个群到监控名单")

    async def _scheduled_moderation(self):
        while True:
            interval = self.config.get("fetch_interval_minutes", 10)
            await asyncio.sleep(interval * 60)
            await self._run_moderation()

    async def _run_moderation(self):
        group_ids = self.config.get("group_ids", [])
        if not group_ids:
            return

        fetch_count = self.config.get("fetch_count", 50)
        system_prompt = self.config.get("moderation_prompt", "").strip()
        if not system_prompt:
            system_prompt = (
                "你是一个群聊管理员。请检查以下消息，判断是否包含违法、色情或辱骂内容。"
                "你必须使用 report_violations 工具上报结果："
                "将违规消息的编号放入 violation_ids 数组，如果没有违规，就传入空数组 []。"
            )

        provider_id = self.config.get("chat_provider", "")
        if not provider_id:
            all_providers = self.context.get_all_providers()
            if all_providers:
                provider_id = all_providers[0].provider_id
            else:
                logger.error("没有可用的 LLM 提供商，审核终止")
                return

        platform = self.context.get_platform(filter.PlatformAdapterType.AIOCQHTTP)
        if not platform:
            logger.error("aiocqhttp 平台未启用")
            return
        client = platform.get_client()

        for gid in group_ids:
            try:
                raw_msgs = await client.api.call_action(
                    "get_group_msg_history", group_id=gid, count=fetch_count
                )
            except Exception as e:
                logger.error(f"获取群 {gid} 的消息失败: {e}")
                continue

            msg_infos = []  # [(msg_id, text, index), ...]
            idx = 1
            for msg in raw_msgs:
                msg_id = msg.get("message_id")
                segments = msg.get("message", [])
                text_parts = []
                for seg in segments:
                    if seg["type"] == "text":
                        text_parts.append(seg["data"]["text"])
                text = " ".join(text_parts).strip()
                if text:
                    msg_infos.append((msg_id, text, idx))
                    idx += 1

            if not msg_infos:
                continue

            prompt_lines = ["群聊消息如下："]
            for _, text, num in msg_infos:
                prompt_lines.append(f"{num}. {text}")
            prompt = "\n".join(prompt_lines)

            try:
                tool = ReportViolationsTool()
                tool_set = ToolSet([tool])  # 使用 ToolSet

                llm_resp = await self.context.tool_loop_agent(
                    event=None,
                    chat_provider_id=provider_id,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    tools=tool_set,
                    max_steps=1,
                    tool_call_timeout=30,
                )
            except Exception as e:
                logger.error(f"LLM 审核请求失败（群 {gid}）: {e}")
                continue

            violation_ids = []
            if llm_resp.tools_call_args:
                call_args = llm_resp.tools_call_args[0]
                violation_ids = call_args.get("violation_ids", [])

            for num in violation_ids:
                try:
                    target = next(
                        (m for m in msg_infos if m[2] == num), None
                    )
                    if target:
                        real_msg_id = target[0]
                        await client.api.call_action(
                            "delete_msg", message_id=real_msg_id
                        )
                        logger.info(f"已撤回违规消息 {real_msg_id}（群 {gid}，编号 {num}）")
                except Exception as e:
                    logger.error(f"撤回消息失败（群 {gid}，编号 {num}）: {e}")