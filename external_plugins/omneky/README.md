# Omneky plugin for Grok Build

Omneky connects Grok to paid media: analyze ROAS/CTR/spend by creative,
campaign, or channel; manage your brand product catalogue; generate image ads
and product videos; and launch or pause campaigns on Meta/Facebook, Google,
LinkedIn, Reddit, and TikTok. Hosted MCP: https://mcp.omneky.com/mcp. New
launches stay paused unless you ask to go live.

## Installation

In Grok Build, open `/plugin`, search for **Omneky**, and install.

On first connection, Grok opens Omneky sign-in in the browser. Use an Omneky
account (email/password or Google). Do not paste an API key or JWT into chat.

## Authentication

The plugin connects only to `https://mcp.omneky.com/mcp`. Authentication is
OAuth 2.1 against that host.

Network endpoints:

- `https://mcp.omneky.com/mcp` — hosted MCP (streamable HTTP)
- `https://mcp.omneky.com/authorize`, `/token`, `/register` — OAuth 2.1 + DCR
- `https://cgp.omneky.com/login` — human sign-in (password / Google)

Credentials: an Omneky account. The access token is an Omneky Nexus JWT used as
`Authorization: Bearer` on `/mcp`. No API key is stored in the plugin. Tools
are scoped to brands the signed-in user can access.

## License

Proprietary. Use of the hosted MCP is governed by Omneky's terms.
