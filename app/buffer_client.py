from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from app.config import AppConfig

logger = logging.getLogger(__name__)

BUFFER_GRAPHQL_ENDPOINT = "https://api.buffer.com"


@dataclass(frozen=True)
class BufferScheduleResult:
    post_id: str
    status: str
    due_at: str
    channel_id: str
    media_url: str


class BufferClient:
    def __init__(self, config: AppConfig, session: requests.Session | None = None) -> None:
        self.config = config
        if not self.config.buffer_api_key:
            raise EnvironmentError("BUFFER_API_KEY environment variable is required.")
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.buffer_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _execute_graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        resp = self.session.post(
            BUFFER_GRAPHQL_ENDPOINT,
            json=payload,
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Buffer GraphQL HTTP {resp.status_code} error: {resp.text[:500]}"
            )

        body = resp.json()
        if "errors" in body and body["errors"]:
            messages = "; ".join(
                str(err.get("message", err)) for err in body["errors"]
            )
            raise RuntimeError(f"Buffer GraphQL top-level error: {messages}")

        data = body.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"Malformed Buffer GraphQL response: {body}")
        return data

    def verify_account_and_channel(self) -> dict[str, Any]:
        query = """
        query VerifyChannel($channelId: ChannelId!) {
          channel(input: { id: $channelId }) {
            id
            name
            service
            organizationId
          }
        }
        """
        data = self._execute_graphql(
            query,
            variables={"channelId": self.config.buffer_channel_id},
        )
        channel = data.get("channel")
        if not channel:
            raise RuntimeError(
                f"Buffer channel '{self.config.buffer_channel_id}' was not found or is inaccessible."
            )

        org_id = channel.get("organizationId")
        if org_id and org_id != self.config.buffer_organization_id:
            raise RuntimeError(
                f"Buffer organization mismatch: expected '{self.config.buffer_organization_id}', "
                f"got '{org_id}'."
            )
        return channel

    def find_existing_scheduled_post(
        self,
        due_at_iso: str,
        caption_prefix: str,
    ) -> BufferScheduleResult | None:
        query = """
        query CheckScheduledPosts($orgId: OrganizationId!, $channelIds: [ChannelId!]) {
          posts(
            input: {
              organizationId: $orgId
              filter: {
                channelIds: $channelIds
                status: [scheduled]
              }
            }
          ) {
            edges {
              node {
                id
                text
                dueAt
                status
              }
            }
          }
        }
        """
        try:
            data = self._execute_graphql(
                query,
                variables={
                    "orgId": self.config.buffer_organization_id,
                    "channelIds": [self.config.buffer_channel_id],
                },
            )
            edges = (data.get("posts") or {}).get("edges") or []
            norm_prefix = caption_prefix.strip()[:60]
            for edge in edges:
                node = (edge or {}).get("node") or {}
                node_due = str(node.get("dueAt") or "")
                node_text = str(node.get("text") or "").strip()
                if node_due == due_at_iso or (norm_prefix and node_text.startswith(norm_prefix)):
                    return BufferScheduleResult(
                        post_id=str(node["id"]),
                        status=str(node.get("status") or "scheduled"),
                        due_at=node_due or due_at_iso,
                        channel_id=self.config.buffer_channel_id,
                        media_url="",
                    )
        except Exception as exc:
            logger.warning("Could not query existing scheduled posts for reconciliation: %s", exc)
        return None

    def schedule_instagram_post(
        self,
        text: str,
        media_url: str,
        media_type: str,
        due_at_iso: str,
        alt_text: str = "",
    ) -> BufferScheduleResult:
        if not text or not text.strip():
            raise ValueError("Post caption text cannot be empty.")
        if not media_url or not media_url.startswith("https://"):
            raise ValueError(f"Media URL must be a valid public HTTPS URL, got: {media_url}")

        parsed_due = datetime.fromisoformat(due_at_iso.replace("Z", "+00:00"))
        now_utc = datetime.now(tz=timezone.utc)
        if parsed_due <= now_utc + timedelta(minutes=1):
            raise ValueError(
                f"Target publication timestamp {due_at_iso} must be in the future (now={now_utc.isoformat()})."
            )

        reconciled = self.find_existing_scheduled_post(
            due_at_iso=due_at_iso,
            caption_prefix=text[:80],
        )
        if reconciled is not None:
            logger.info(
                "Reconciled existing scheduled Buffer post id=%s for dueAt=%s",
                reconciled.post_id,
                due_at_iso,
            )
            return BufferScheduleResult(
                post_id=reconciled.post_id,
                status=reconciled.status,
                due_at=reconciled.due_at,
                channel_id=self.config.buffer_channel_id,
                media_url=media_url,
            )

        is_reel = media_type.upper() == "REEL"
        if is_reel:
            assets_input: dict[str, Any] = {
                "videos": [{"url": media_url}],
            }
            ig_metadata: dict[str, Any] = {
                "type": "reel",
                "shouldShareToFeed": True,
            }
        else:
            image_entry: dict[str, Any] = {"url": media_url}
            if alt_text:
                image_entry["metadata"] = {"altText": alt_text[:220]}
            assets_input = {
                "images": [image_entry],
            }
            ig_metadata = {
                "type": "post",
                "shouldShareToFeed": True,
            }

        mutation = """
        mutation ScheduleInstagramPost($input: CreatePostInput!) {
          createPost(input: $input) {
            __typename
            ... on PostActionSuccess {
              post {
                id
                status
                dueAt
                text
              }
            }
            ... on MutationError {
              message
            }
          }
        }
        """

        variables = {
            "input": {
                "channelId": self.config.buffer_channel_id,
                "text": text,
                "schedulingType": "automatic",
                "mode": "customScheduled",
                "dueAt": due_at_iso,
                "assets": assets_input,
                "metadata": {
                    "instagram": ig_metadata,
                },
            }
        }

        data = self._execute_graphql(mutation, variables=variables)
        result = data.get("createPost") or {}
        typename = result.get("__typename")

        if typename == "MutationError":
            err_msg = result.get("message") or "Unknown Buffer MutationError"
            raise RuntimeError(f"Buffer createPost returned MutationError: {err_msg}")

        if typename != "PostActionSuccess" or not result.get("post"):
            raise RuntimeError(f"Unexpected Buffer createPost response payload: {result}")

        post_obj = result["post"]
        post_id = str(post_obj.get("id") or "")
        status = str(post_obj.get("status") or "scheduled")
        returned_due_at = str(post_obj.get("dueAt") or due_at_iso)

        if not post_id:
            raise RuntimeError(f"Buffer createPost succeeded without returning a post ID: {result}")

        return BufferScheduleResult(
            post_id=post_id,
            status=status,
            due_at=returned_due_at,
            channel_id=self.config.buffer_channel_id,
            media_url=media_url,
        )
