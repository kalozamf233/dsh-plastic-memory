// ============================================================================
// dsh-plastic-memory —— client half（浏览器 bundle）
// ----------------------------------------------------------------------------
// 「可塑性记忆」大脑仪表盘：
//   1. 左侧栏入口（sidebar.panellist）：大脑图标「可塑性记忆」；
//   2. 主面板（main）：
//      · 未初始化 → 工作区路径输入 + 初始化（必须由用户指定文件夹）
//      · 已初始化 → 实时仪表盘（2s 轮询 /plastic-memory/status）：
//        双网络指标 / 分区神经元 / 情绪构建 / 人格特质 / 15 种分子浓度 /
//        操控网络决策与结构变更日志 / 听·说·读·写·冻结 操作区
// 零平台依赖：仅 require("react") / require("react/jsx-runtime")。
// ============================================================================
window.__ModuleLoader__.load({
	id: "dsh-plastic-memory",
	factory: (require) => {
		var module = { exports: {} };
		var exports = module.exports;
		var react = require("react");
		var jsxRuntime = require("react/jsx-runtime");
		var jsx = jsxRuntime.jsx;
		var jsxs = jsxRuntime.jsxs;
		var useEffect = react.useEffect;
		var useState = react.useState;
		var useRef = react.useRef;

		var PANEL_ID = "plastic-memory";
		var ROUTE = "/plastic-memory";
		var TITLE = "可塑性记忆";

		var CSS = [
			".pm-root{position:absolute;top:0;left:0;right:0;bottom:0;overflow-y:auto;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.35) transparent;}",
			".pm-root::-webkit-scrollbar{width:9px;height:9px;}",
			".pm-root::-webkit-scrollbar-thumb{background:rgba(255,255,255,.28);border-radius:5px;}",
			".pm-root::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.45);}",
			".pm-root::-webkit-scrollbar-track{background:transparent;}",
			".pm-inner{max-width:860px;margin:0 auto;padding:24px 22px 48px;display:flex;flex-direction:column;gap:12px;}",
			".pm-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;}",
			".pm-head .pm-title{font-size:17px;font-weight:600;color:var(--dsw-alias-text-1,#e8eaf0);}",
			".pm-head .pm-desc{font-size:12px;color:var(--dsw-alias-text-3,rgba(232,234,240,.5));}",
			".pm-badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;border-radius:999px;padding:2px 10px;border:1px solid;}",
			".pm-badge.train{color:#3fbf6e;border-color:rgba(63,191,110,.45);background:rgba(63,191,110,.1);}",
			".pm-badge.frozen{color:#4e6ef2;border-color:rgba(78,110,242,.45);background:rgba(78,110,242,.12);}",
			".pm-badge.off{color:var(--dsw-alias-text-3,rgba(232,234,240,.5));border-color:var(--dsw-alias-border-l3,rgba(255,255,255,.1));background:var(--dsw-alias-bg-layer-2,rgba(255,255,255,.04));}",
			".pm-card{background:var(--dsw-alias-bg-layer-2,rgba(255,255,255,.035));border:1px solid var(--dsw-alias-border-l3,rgba(255,255,255,.09));border-radius:12px;padding:14px 16px;}",
			".pm-card h2{margin:0 0 10px;font-size:13px;font-weight:600;color:var(--dsw-alias-text-1,#e8eaf0);letter-spacing:.2px;display:flex;align-items:center;justify-content:space-between;gap:8px;}",
			".pm-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;}",
			".pm-stat{background:var(--dsw-alias-bg-layer-3,rgba(0,0,0,.2));border-radius:10px;padding:10px 12px;}",
			".pm-stat .v{font-size:16px;font-weight:600;color:var(--dsw-alias-text-1,#e8eaf0);font-variant-numeric:tabular-nums;}",
			".pm-stat .k{font-size:11px;color:var(--dsw-alias-text-3,rgba(232,234,240,.5));margin-top:3px;}",
			".pm-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;}",
			".pm-bar-row .nm{width:92px;font-size:11px;color:var(--dsw-alias-text-2,rgba(232,234,240,.75));text-align:right;flex-shrink:0;}",
			".pm-bar-row .br{flex:1;height:7px;border-radius:4px;background:var(--dsw-alias-bg-layer-3,rgba(0,0,0,.22));overflow:hidden;}",
			".pm-bar-row .fl{height:100%;border-radius:4px;background:var(--dsw-alias-brand-primary,#4e6ef2);transition:width .5s ease;}",
			".pm-bar-row .nm2{width:34px;font-size:10px;color:var(--dsw-alias-text-3,rgba(232,234,240,.5));font-variant-numeric:tabular-nums;}",
			".pm-mols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:2px 14px;}",
			".pm-emo{display:flex;gap:6px;flex-wrap:wrap;}",
			".pm-emo .chip{font-size:11px;border-radius:6px;padding:3px 8px;background:var(--dsw-alias-bg-layer-3,rgba(0,0,0,.22));border:1px solid var(--dsw-alias-border-l3,rgba(255,255,255,.09));color:var(--dsw-alias-text-2,rgba(232,234,240,.75));}",
			".pm-emo .chip.hot{color:#ffd08a;border-color:rgba(255,178,89,.5);background:rgba(255,178,89,.12);}",
			".pm-actions{display:flex;gap:8px;flex-wrap:wrap;}",
			".pm-btn{border:1px solid transparent;border-radius:8px;padding:8px 14px;font-size:12.5px;font-weight:500;cursor:pointer;font-family:inherit;transition:background .12s ease,opacity .12s ease;}",
			".pm-btn:disabled{opacity:.45;cursor:not-allowed;}",
			".pm-btn.ghost{background:transparent;border-color:var(--dsw-alias-border-l3,rgba(255,255,255,.14));color:var(--dsw-alias-text-1,#e8eaf0);}",
			".pm-btn.ghost:hover:not(:disabled){background:var(--dsw-alias-interactive-bg-hover,rgba(255,255,255,.06));}",
			".pm-btn.solid{background:var(--dsw-alias-brand-primary,#4e6ef2);color:#fff;}",
			".pm-btn.solid:hover:not(:disabled){filter:brightness(1.1);}",
			".pm-btn.danger{background:transparent;border-color:rgba(229,72,77,.5);color:#e5484d;}",
			".pm-btn.danger:hover:not(:disabled){background:rgba(229,72,77,.1);}",
			".pm-input{box-sizing:border-box;width:100%;background:var(--dsw-alias-bg-layer-3,rgba(0,0,0,.22));border:1px solid var(--dsw-alias-border-l3,rgba(255,255,255,.09));border-radius:8px;padding:8px 10px;font-size:13px;color:var(--dsw-alias-text-1,#e8eaf0);font-family:inherit;outline:none;}",
			".pm-input:focus{border-color:var(--dsw-alias-brand-primary,#4e6ef2);}",
			".pm-msg{font-size:12.5px;line-height:1.6;border-radius:10px;padding:10px 12px;border:1px solid;}",
			".pm-msg.ok{color:var(--dsw-alias-state-success-primary,#3fbf6e);border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary,#3fbf6e) 40%,transparent);background:color-mix(in srgb,var(--dsw-alias-state-success-primary,#3fbf6e) 10%,transparent);}",
			".pm-msg.err{color:var(--dsw-alias-state-error-primary,#e5484d);border-color:color-mix(in srgb,var(--dsw-alias-state-error-primary,#e5484d) 40%,transparent);background:color-mix(in srgb,var(--dsw-alias-state-error-primary,#e5484d) 10%,transparent);}",
			".pm-log{font-size:11.5px;color:var(--dsw-alias-text-3,rgba(232,234,240,.55));font-family:ui-monospace,Consolas,monospace;line-height:1.7;max-height:130px;overflow-y:auto;}",
			".pm-speak{border-radius:10px;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary,#4e6ef2) 35%,transparent);background:color-mix(in srgb,var(--dsw-alias-brand-primary,#4e6ef2) 8%,transparent);padding:12px 14px;font-size:13px;line-height:1.7;color:var(--dsw-alias-text-1,#e8eaf0);}",
			".pm-tabs{display:flex;gap:4px;padding:10px 16px 0;border-bottom:1px solid var(--dsw-alias-border-l3,rgba(255,255,255,.08));flex-shrink:0;}",
			".pm-tab{padding:7px 16px;font-size:12.5px;font-weight:500;color:var(--dsw-alias-text-3,rgba(232,234,240,.55));cursor:pointer;border:1px solid transparent;border-bottom:none;border-radius:8px 8px 0 0;background:transparent;}",
			".pm-tab:hover{color:var(--dsw-alias-text-1,#e8eaf0);}",
			".pm-tab.active{color:var(--dsw-alias-brand-primary,#4e6ef2);border-color:var(--dsw-alias-border-l3,rgba(255,255,255,.1));background:var(--dsw-alias-bg-layer-2,rgba(255,255,255,.04));}",
			".pm-wake{position:fixed;top:18px;right:18px;z-index:99;display:flex;align-items:center;gap:8px;background:color-mix(in srgb,var(--dsw-alias-state-success-primary,#3fbf6e) 14%,var(--dsw-alias-bg-layer-2,rgba(30,32,40,.96)));border:1px solid color-mix(in srgb,var(--dsw-alias-state-success-primary,#3fbf6e) 45%,transparent);border-radius:12px;padding:12px 16px;font-size:13px;color:var(--dsw-alias-text-1,#e8eaf0);box-shadow:0 8px 28px rgba(0,0,0,.4);animation:pm-wake-in .35s ease;}",
			".pm-wake b{color:var(--dsw-alias-state-success-primary,#3fbf6e);}",
			"@keyframes pm-wake-in{from{opacity:0;transform:translateY(-10px);}to{opacity:1;transform:translateY(0);}}",
		].join("\n");

		function ensureCss() {
			if (typeof document === "undefined") return;
			if (document.querySelector("style[data-pm-css]")) return;
			var tag = document.createElement("style");
			tag.setAttribute("data-pm-css", "1");
			tag.textContent = CSS;
			document.head.appendChild(tag);
		}

		function Icon(props) {
			var size = props && typeof props.size === "number" ? props.size : 18;
			var active = !!(props && props.active);
			var color = active ? "var(--dsw-alias-brand-primary,#4e6ef2)" : "currentColor";
			return jsx("svg", {
				width: size,
				height: size,
				viewBox: "0 0 18 18",
				fill: "none",
				"aria-hidden": "true",
				children: jsxs("g", {
					stroke: color,
					strokeWidth: 1.4,
					strokeLinejoin: "round",
					children: [
						jsx("path", { d: "M9 2.2c-2.4-1.1-4.9-.8-6.3.4C3 3.6 3.4 4.7 3.4 6.2c0 3.4 1.9 6.9 5.6 9.6 3.7-2.7 5.6-6.2 5.6-9.6 0-1.5.4-2.6.7-3.6-1.4-1.2-3.9-1.5-6.3-.4Z" }),
						jsx("path", { d: "M5.6 7.2h6.8M5.6 9.2h6.8M6.8 11.2h4.4" }),
					],
				}),
			});
		}

		function postJson(path, payload) {
			return fetch(ROUTE + path, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify(payload || {}),
			}).then(function (r) {
				return r.json().catch(function () {
					return { ok: false, error: "响应解析失败" };
				});
			});
		}
		function getJson(path) {
			return fetch(ROUTE + path)
				.then(function (r) { return r.json(); })
				.catch(function (e) {
					return { ok: false, error: String((e && e.message) || e) };
				});
		}

		function fmt(n) {
			if (n === null || n === undefined || !Number.isFinite(Number(n))) return "—";
			var v = Number(n);
			if (Math.abs(v) >= 10000) return (v / 10000).toFixed(1) + "万";
			if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + "k";
			return String(Math.round(v * 100) / 100);
		}

		var EMOTION_LABELS = { valence: "效价", arousal: "唤醒", joy: "喜悦", sadness: "悲伤", anger: "愤怒", fear: "恐惧", disgust: "厌恶", surprise: "惊讶", calm: "平静", anxiety: "焦虑", trust: "信任", love: "爱" };

		// ---------------------------------------------------------------------
		// 主面板
		// ---------------------------------------------------------------------
		function PanelPage(props, ctx) {
			var statusRef = props ? props.statusRef : null;
			var status = statusRef ? statusRef.current : null;
			var setMsg = props ? props.setMsg : null;
			var busy = props ? props.busy : "";
			var setBusy = props ? props.setBusy : null;
			var hearText = props ? props.hearText : "";
			var setHearText = props ? props.setHearText : null;
			var queryText = props ? props.queryText : "";
			var setQueryText = props ? props.setQueryText : null;
			var speakText = props ? props.speakText : "";
			var wsPath = props ? props.wsPath : "";
			var setWsPath = props ? props.setWsPath : null;
			var initPath = props ? props.initPath : "";
			var setInitPath = props ? props.setInitPath : null;
			var extPath = props ? props.extPath : "";
			var setExtPath = props ? props.setExtPath : null;

			if (!status || !status.initialized) {
				return jsxs("div", { className: "pm-root", children: [
					jsx("div", { className: "pm-inner", children: [
						jsxs("div", { className: "pm-card", children: [
							jsx("h2", { children: "初始化：指定工作区文件夹" }),
							jsx("p", { style: { margin: "0 0 12px", fontSize: 12.5, lineHeight: 1.7, color: "var(--dsw-alias-text-2,rgba(232,234,240,.75))" }, children: "可塑性记忆的初始化必须由用户指定一个文件夹作为工作区：操控神经网络本体与记忆神经网络的权重文件、感知文件夹（senses/）都会建立在该文件夹中。" }),
							jsxs("div", { style: { display: "flex", gap: 8 }, children: [
								jsx("input", { className: "pm-input", placeholder: "工作区绝对路径，例如 E:\\digital-life\\memory-1", value: initPath, spellCheck: false, onChange: function (e) { setInitPath(e.target.value); } }),
								jsx("button", { type: "button", className: "pm-btn solid", disabled: busy !== "", onClick: function () { onInit(); }, children: busy === "init" ? "初始化中…" : "初始化并启动" }),
							] }),
						] }),
						jsxs("div", { className: "pm-card", children: [
							jsx("h2", { children: "或恢复已有工作区" }),
							jsxs("div", { style: { display: "flex", gap: 8 }, children: [
								jsx("input", { className: "pm-input", placeholder: "已有工作区路径", value: wsPath, spellCheck: false, onChange: function (e) { setWsPath(e.target.value); } }),
								jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onLoad(); }, children: busy === "load" ? "加载中…" : "加载工作区" }),
							] }),
						] }),
					] }),
				] });

				function onInit() {
					if (!initPath.trim()) { setMsg({ kind: "err", text: "必须由用户指定一个工作区文件夹" }); return; }
					setBusy("init");
					postJson("/init", { workspace: initPath.trim() }).then(function (r) {
						setBusy("");
						if (r && r.ok) setMsg({ kind: "ok", text: "工作区已初始化并启动：" + r.workspace });
						else setMsg({ kind: "err", text: (r && r.error) || "初始化失败" });
					});
				}
				function onLoad() {
					if (!wsPath.trim()) { setMsg({ kind: "err", text: "请填写工作区路径" }); return; }
					setBusy("load");
					postJson("/load", { workspace: wsPath.trim() }).then(function (r) {
						setBusy("");
						if (r && r.ok) setMsg({ kind: "ok", text: "已加载工作区：" + r.workspace });
						else setMsg({ kind: "err", text: (r && r.error) || "加载失败" });
					});
				}
			}

			var s = status;
			var emo = s.emotion || {};
			var persona = s.persona || {};
			var traits = persona.traits || {};
			var moral = persona.moralFoundations || {};
			var mols = s.endocrine && s.endocrine.molecules ? s.endocrine.molecules : {};
			var ctrl = s.controller || {};
			var mem = s.memory || {};
			var extList = Array.isArray(s.extensions) ? s.extensions : [];
			var changes = (mem.structuralChanges || []).slice(-8).reverse();
			var MORAL_ZH = { care: "关爱", fairness: "公平", loyalty: "忠诚", authority: "权威", sanctity: "圣洁" };
			var hotEmotions = [];
			Object.keys(emo).forEach(function (k) {
				if (k === "valence" || k === "arousal") return;
				if (emo[k] > 0.35) hotEmotions.push(EMOTION_LABELS[k] || k);
			});

			return jsx("div", { className: "pm-root", children: jsxs("div", { className: "pm-inner", children: [
				jsxs("div", { className: "pm-head", children: [
					jsx("span", { className: "pm-title", children: "可塑性记忆 · " + (s.name || "数字生命") }),
					jsx("span", { className: "pm-badge " + (s.mode === "training" ? "train" : "frozen"), children: s.mode === "training" ? "● 训练态" : "❄ 冻结态" }),
					jsx("span", { className: "pm-badge " + (s.running ? "train" : "off"), children: s.running ? "心跳运行中" : "心跳停止" }),
					s.sleep && s.sleep.asleep
						? jsx("span", { className: "pm-badge off", children: "🌙 休眠中（说话可唤醒）" })
						: jsx("span", { className: "pm-badge train", children: "☀ 清醒 " + Math.round(((s.sleep && s.sleep.wakefulness) || 1) * 100) + "%" }),
				] }),
				jsx("div", { className: "pm-desc", style: { fontSize: 11.5, color: "var(--dsw-alias-text-3,rgba(232,234,240,.5))", wordBreak: "break-all" }, children: s.workspace }),

				jsxs("div", { className: "pm-stats", children: [
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: fmt(mem.paramCount) }), jsx("div", { className: "k", children: "记忆网络参数" })] }),
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: (mem.layerSizes || []).join(" / ") }), jsx("div", { className: "k", children: "分区层/联想层/输出层神经元" })] }),
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: mem.predErrorEma !== undefined ? Number(mem.predErrorEma).toFixed(4) : "—" }), jsx("div", { className: "k", children: "预测误差 EMA（越低越熟悉）" })] }),
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: fmt(mem.observeCount) }), jsx("div", { className: "k", children: "感知注入次数" })] }),
				] }),

				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: jsxs("span", { children: ["构建中的情绪", hotEmotions.length ? " · " + hotEmotions.join(" / ") : ""] }) }),
					jsx("div", { className: "pm-emo", children: Object.keys(emo).map(function (k) {
						return jsx("span", { key: k, className: "chip" + (emo[k] > 0.35 ? " hot" : ""), children: (EMOTION_LABELS[k] || k) + " " + Number(emo[k]).toFixed(2) });
					}) }),
				] }),

				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: "人格特质（叙事自我）" }),
					jsx("div", { style: { marginBottom: 8 }, children: Object.keys(traits).slice(0, 14).map(function (k) {
						var v = Number(traits[k]);
						return jsxs("div", { key: k, className: "pm-bar-row", children: [
							jsx("span", { className: "nm", children: k }),
							jsx("div", { className: "br", children: jsx("div", { className: "fl", style: { width: Math.max(2, Math.min(100, (v + 1) * 50)) + "%", background: v >= 0 ? "var(--dsw-alias-brand-primary,#4e6ef2)" : "#e5484d" } }) }),
							jsx("span", { className: "nm2", children: v.toFixed(2) }),
						] });
					}) }),
					jsx("div", { style: { fontSize: 11, color: "var(--dsw-alias-text-3,rgba(232,234,240,.5))", marginBottom: 4 }, children: "道德基础五原则" }),
					jsx("div", { children: Object.keys(moral).map(function (k) {
						var v = Number(moral[k]);
						return jsxs("div", { key: k, className: "pm-bar-row", children: [
							jsx("span", { className: "nm", children: MORAL_ZH[k] || k }),
							jsx("div", { className: "br", children: jsx("div", { className: "fl", style: { width: Math.max(2, Math.min(100, (v + 1) * 50)) + "%" } }) }),
							jsx("span", { className: "nm2", children: v.toFixed(2) }),
						] });
					}) }),
				] }),

				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: "内分泌 · 15 种分子" }),
					jsx("div", { className: "pm-mols", children: Object.keys(mols).map(function (k) {
						var m = mols[k] || {};
						return jsxs("div", { key: k, className: "pm-bar-row", style: { marginBottom: 4 }, children: [
							jsx("span", { className: "nm", children: m.name || k }),
							jsx("div", { className: "br", children: jsx("div", { className: "fl", style: { width: Math.max(2, Math.min(100, (m.level || 0) * 100)) + "%", background: "color-mix(in srgb,var(--dsw-alias-brand-primary,#4e6ef2) 70%,#ffb259)" } }) }),
							jsx("span", { className: "nm2", children: (m.level || 0).toFixed(2) }),
						] });
					}) }),
				] }),

				jsxs("div", { className: "pm-stats", children: [
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: fmt(ctrl.decisionCount) }), jsx("div", { className: "k", children: "操控网络决策" })] }),
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: "+" + (ctrl.growCount || 0) + " / −" + (ctrl.pruneCount || 0) }), jsx("div", { className: "k", children: "增/删神经元" })] }),
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: ctrl.rewardEma !== undefined ? Number(ctrl.rewardEma).toFixed(3) : "—" }), jsx("div", { className: "k", children: "元学习奖励 EMA" })] }),
					jsxs("div", { className: "pm-stat", children: [jsx("div", { className: "v", children: ctrl.lrMod !== undefined ? Number(ctrl.lrMod).toFixed(2) + "×" : "—" }), jsx("div", { className: "k", children: "操控网络学习率调制" })] }),
				] }),

				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: "对话（身体 → 灵魂）" }),
					jsx("div", { style: { display: "flex", gap: 8, marginBottom: 8 }, children: [
						jsx("input", { className: "pm-input", placeholder: "对它说点什么（听：经操控网络编码后注入记忆网络）", value: hearText, spellCheck: false, onChange: function (e) { setHearText(e.target.value); } }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onHear(); }, children: "听" }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onSpeak(); }, children: "说" }),
					] }),
					jsx("div", { style: { display: "flex", gap: 8 }, children: [
						jsx("input", { className: "pm-input", placeholder: "联想检索记忆（读）", value: queryText, spellCheck: false, onChange: function (e) { setQueryText(e.target.value); } }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onRead(); }, children: "读" }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onWrite(); }, children: "写（强化刻写）" }),
					] }),
					speakText ? jsx("div", { className: "pm-speak", children: speakText }) : null,
				] }),

				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: "生命周期" }),
					jsx("div", { className: "pm-actions", children: [
						s.mode === "training"
							? jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onCtrl("freeze"); }, children: "❄ 固定权重（freeze）" })
							: jsx("button", { type: "button", className: "pm-btn solid", disabled: busy !== "", onClick: function () { onCtrl("thaw"); }, children: "▶ 恢复训练（thaw）" }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onCtrl("save"); }, children: "保存权重" }),
						s.sleep && s.sleep.asleep
							? jsx("button", { type: "button", className: "pm-btn solid", disabled: busy !== "", onClick: function () { onCtrl("wake"); }, children: "☀ 唤醒" })
							: jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onCtrl("sleep"); }, children: "🌙 休眠" }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onCtrl("grow"); }, children: "+ 增神经元" }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onCtrl("prune"); }, children: "− 剪神经元" }),
						jsx("button", { type: "button", className: "pm-btn danger", disabled: busy !== "", onClick: function () { onCtrl("reset"); }, children: "返祖（重置为出厂基因）" }),
					] }),
				] }),

				changes.length
					? jsxs("div", { className: "pm-card", children: [
						jsx("h2", { children: "结构变更（操控网络）" }),
						jsx("div", { className: "pm-log", children: changes.map(function (c) {
							return jsx("div", { key: String(c.tick) + c.kind + c.layer, children: "tick " + c.tick + "  " + (c.kind === "grow" ? "▲ 层" + c.layer + " 新增神经元 #" + c.index + "（分化自 #" + c.source + "）" : "▼ 层" + c.layer + " 剪除神经元 #" + c.index) });
						}) }),
					] })
					: null,
			] }) });

			function onHear() {
				if (!hearText.trim()) return;
				setBusy("hear");
				postJson("/controller", { action: "hear", payload: { text: hearText } }).then(function (r) {
					setBusy("");
					if (r && r.ok) {
						setHearText("");
						setMsg({ kind: "ok", text: "已听见（预测误差 " + (r.result && r.result.predError) + "）" });
					} else setMsg({ kind: "err", text: (r && r.error) || "失败" });
				});
			}
			function onSpeak() {
				setBusy("speak");
				postJson("/controller", { action: "speak", payload: { text: hearText } }).then(function (r) {
					setBusy("");
					if (r && r.ok && r.result && typeof r.result.text === "string") {
						setHearText("");
						setSpeakText(r.result.text);
					} else setMsg({ kind: "err", text: (r && r.error) || "失败" });
				});
			}
			function onRead() {
				if (!queryText.trim()) return;
				setBusy("read");
				postJson("/controller", { action: "read", payload: { query: queryText } }).then(function (r) {
					setBusy("");
					if (r && r.ok) {
						setQueryText("");
						var n = r.result && r.result.matches ? r.result.matches.length : 0;
						setMsg({ kind: "ok", text: "检索到 " + n + " 条相关记忆" + (n ? "（详见工具 plastic_memory_read）" : "") });
					} else setMsg({ kind: "err", text: (r && r.error) || "失败" });
				});
			}
			function onWrite() {
				if (!hearText.trim()) { setMsg({ kind: "err", text: "先在输入框填写要写入的内容" }); return; }
				setBusy("write");
				postJson("/controller", { action: "write", payload: { text: hearText, module: "persona", epochs: 8 } }).then(function (r) {
					setBusy("");
					if (r && r.ok) {
						setHearText("");
						setMsg({ kind: "ok", text: "已刻写进人格（epochs=" + (r.result && r.result.epochs) + "）" });
					} else setMsg({ kind: "err", text: (r && r.error) || "失败" });
				});
			}
			function onCtrl(action) {
				setBusy(action);
				postJson("/controller", { action: action, payload: {} }).then(function (r) {
					setBusy("");
					if (r && r.ok) setMsg({ kind: "ok", text: action + " 完成" });
					else setMsg({ kind: "err", text: (r && r.error) || "失败" });
				});
			}
		}

		// ---------------------------------------------------------------------
		// 模型构建子页面（合并自 AI 模型构建插件，改为自动生成）
		// ---------------------------------------------------------------------
		function BuilderPage(props) {
			var setMsg = props.setMsg;
			var busy = props.busy;
			var setBusy = props.setBusy;
			var genPath = props.genPath;
			var setGenPath = props.setGenPath;
			var extPath = props.extPath;
			var setExtPath = props.setExtPath;
			var extList = Array.isArray(props.extList) ? props.extList : [];

			return jsx("div", { className: "pm-root", children: jsxs("div", { className: "pm-inner", children: [
				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: "自动生成神经网络" }),
					jsx("p", { style: { margin: "0 0 12px", fontSize: 12.5, lineHeight: 1.7, color: "var(--dsw-alias-text-2,rgba(232,234,240,.75))" }, children: "一键生成与记忆神经网络同类型的全新网络（可塑性记忆 .brain.json 格式），生成后自动挂载为外部扩容，随感知一起训练。不再需要手动配置层数/参数——网络按记忆神经网络的标准架构自动初始化。" }),
					jsx("div", { style: { display: "flex", gap: 8, marginBottom: 10 }, children: [
						jsx("input", { className: "pm-input", placeholder: "网络保存路径（不存在则自动创建），如 E:\\memory-ext\\auto-1.brain.json", value: genPath, spellCheck: false, onChange: function (e) { setGenPath(e.target.value); } }),
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onGenerate(); }, children: busy === "generate" ? "生成中…" : "自动生成并挂载" }),
					] }),
				] }),
				jsxs("div", { className: "pm-card", children: [
					jsx("h2", { children: "神经网络扩展（外部扩容 · 挂载列表）" }),
					jsx("textarea", { className: "pm-input", rows: 3, style: { resize: "vertical", lineHeight: 1.5 }, placeholder: "每行一个路径（可一次挂载多个）。\n填「不存在的路径」→ 自动创建全新的空白网络，例如：\nE:\\memory-ext\\ext1.brain.json\nE:\\memory-ext\\ext2.brain.json", value: extPath, spellCheck: false, onChange: function (e) { setExtPath(e.target.value); } }),
					jsx("div", { style: { display: "flex", gap: 8, margin: "8px 0 10px" }, children: [
						jsx("button", { type: "button", className: "pm-btn ghost", disabled: busy !== "", onClick: function () { onMountExt(); }, children: "挂载（支持多行）" }),
					] }),
					extList.length
						? jsx("div", { children: extList.map(function (ext) {
							return jsxs("div", { key: ext.path, style: { display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 8, background: "var(--dsw-alias-bg-layer-3,rgba(0,0,0,.2))", marginBottom: 6 }, children: [
								jsx("div", { style: { flex: 1, minWidth: 0 }, children: [
									jsx("div", { style: { fontSize: 11.5, color: "var(--dsw-alias-text-2,rgba(232,234,240,.75))", wordBreak: "break-all", fontFamily: "ui-monospace,Consolas,monospace" }, children: ext.path }),
									jsx("div", { style: { fontSize: 11, color: "var(--dsw-alias-text-3,rgba(232,234,240,.5))", marginTop: 3 }, children: "神经元 " + (ext.layerSizes || []).join("/") + " · 参数 " + fmt(ext.paramCount) + " · 误差 " + ext.predErrorEma + " · 感知 " + ext.observeCount }),
								] }),
								jsx("button", { type: "button", className: "pm-btn danger", style: { padding: "4px 10px", fontSize: 11.5 }, disabled: busy !== "", onClick: function () { onUnmountExt(ext.path); }, children: "卸载" }),
							] });
						}) })
						: jsx("div", { style: { fontSize: 11.5, color: "var(--dsw-alias-text-3,rgba(232,234,240,.5))", lineHeight: 1.7 }, children: "暂无扩展。每行填一个「不存在的路径」（如 E:\\memory-ext\\ext1.brain.json）挂载后会自动创建空白网络并开始随感知一起训练。注意：只能挂载本插件的 .brain.json 格式文件，PyTorch 的 model.py / params.json / architecture.json 不能挂载。" }),
				] }),
			] }) });

			function onGenerate() {
				if (!genPath.trim()) { setMsg({ kind: "err", text: "请填写网络保存路径" }); return; }
				setBusy("generate");
				postJson("/controller", { action: "generateNetwork", payload: { path: genPath.trim() } }).then(function (r) {
					setBusy("");
					if (r && r.ok) {
						var g = r.result || {};
						setGenPath("");
						setMsg({ kind: "ok", text: "已生成并挂载：" + (g.path || "") + "（参数 " + (g.paramCount || "?") + "）" });
					} else setMsg({ kind: "err", text: (r && r.error) || "生成失败" });
				});
			}
			function onMountExt() {
				var lines = extPath.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean);
				if (lines.length === 0) { setMsg({ kind: "err", text: "请每行填写一个扩展网络的保存路径" }); return; }
				setBusy("mount");
				postJson("/controller", { action: "mountExtension", payload: { paths: lines } }).then(function (r) {
					setBusy("");
					if (r && r.ok) {
						var res = r.result || {};
						var mounted = (res.mounted || []).length;
						var created = (res.mounted || []).filter(function (x) { return x && x.created; }).length;
						var failed = (res.errors || []).length;
						var text = "挂载成功 " + mounted + " 个（新建 " + created + " 个）";
						if (failed > 0) {
							text += "，失败 " + failed + " 个：" + (res.errors && res.errors[0] ? res.errors[0].error : "");
							setMsg({ kind: "err", text: text });
						} else {
							setExtPath("");
							setMsg({ kind: "ok", text: text });
						}
					} else setMsg({ kind: "err", text: (r && r.error) || "挂载失败" });
				});
			}
			function onUnmountExt(path) {
				setBusy("unmount");
				postJson("/controller", { action: "unmountExtension", payload: { path: path } }).then(function (r) {
					setBusy("");
					if (r && r.ok) setMsg({ kind: "ok", text: "已卸载扩展（权重已保存到原路径）" });
					else setMsg({ kind: "err", text: (r && r.error) || "卸载失败" });
				});
			}
		}

		// ---------------------------------------------------------------------
		// 外层容器：轮询状态 + 消息
		// ---------------------------------------------------------------------
		function Root(ctx) {
			var statusRef = useRef(null);
			var statusValue = statusRef.current;
			var pollRef = useRef(false);
			var msgState = useState(null);
			var msg = msgState[0];
			var setMsg = msgState[1];
			var busyState = useState("");
			var busy = busyState[0];
			var setBusy = busyState[1];
			var hearState = useState("");
			var hearText = hearState[0];
			var setHearText = hearState[1];
			var queryState = useState("");
			var queryText = queryState[0];
			var setQueryText = queryState[1];
			var speakState = useState("");
			var speakText = speakState[0];
			var setSpeakText = speakState[1];
			var wsState = useState("");
			var wsPath = wsState[0];
			var setWsPath = wsState[1];
			var initState = useState("");
			var initPath = initState[0];
			var setInitPath = initState[1];
			var extState = useState("");
			var extPath = extState[0];
			var setExtPath = extState[1];
			var tickState = useState(0);
			var setTick = tickState[1];
			var tabState = useState("dashboard");
			var tab = tabState[0];
			var setTab = tabState[1];
			var genPathState = useState("");
			var genPath = genPathState[0];
			var setGenPath = genPathState[1];
			var wakeToastState = useState(null);
			var wakeToast = wakeToastState[0];
			var setWakeToast = wakeToastState[1];
			var lastWakeAtRef = useRef(null);
			var status = statusRef.current;

			useEffect(function () {
				var timer = setInterval(function () {
					getJson("/status").then(function (r) {
						if (r && r.ok) {
							statusRef.current = r;
							// 苏醒事件检测：sleep.lastWakeAt 变化 → 右上角弹提示
							if (r.sleep && r.sleep.lastWakeAt) {
								if (lastWakeAtRef.current !== null && lastWakeAtRef.current !== r.sleep.lastWakeAt) {
									setWakeToast({ at: r.sleep.lastWakeAt, count: r.sleep.sleepCount });
									setTimeout(function () { setWakeToast(null); }, 6000);
								}
								lastWakeAtRef.current = r.sleep.lastWakeAt;
							}
						}
						setTick(function (t) { return t + 1; });
					});
				}, 2000);
				getJson("/status").then(function (r) {
					if (r && r.ok) {
						statusRef.current = r;
						if (r.sleep && r.sleep.lastWakeAt) lastWakeAtRef.current = r.sleep.lastWakeAt;
					}
					setTick(function (t) { return t + 1; });
				});
				return function () { clearInterval(timer); };
			}, []);

			return jsxs("div", { style: { height: "100%", position: "relative", overflow: "hidden", boxSizing: "border-box", display: "flex", flexDirection: "column" }, children: [
				jsxs("div", { className: "pm-tabs", children: [
					jsx("div", { className: "pm-tab" + (tab === "dashboard" ? " active" : ""), onClick: function () { setTab("dashboard"); }, children: "大脑仪表盘" }),
					jsx("div", { className: "pm-tab" + (tab === "builder" ? " active" : ""), onClick: function () { setTab("builder"); }, children: "模型构建" }),
				] }),
				jsx("div", { style: { flex: 1, position: "relative", overflow: "hidden" }, children:
					tab === "dashboard"
						? PanelPage({
							statusRef: statusRef,
							setMsg: setMsg,
							busy: busy,
							setBusy: setBusy,
							hearText: hearText,
							setHearText: setHearText,
							queryText: queryText,
							setQueryText: setQueryText,
							speakText: speakText,
							setSpeakText: setSpeakText,
							wsPath: wsPath,
							setWsPath: setWsPath,
							initPath: initPath,
							setInitPath: setInitPath,
							extPath: extPath,
							setExtPath: setExtPath,
						})
						: BuilderPage({
							setMsg: setMsg,
							busy: busy,
							setBusy: setBusy,
							genPath: genPath,
							setGenPath: setGenPath,
							extPath: extPath,
							setExtPath: setExtPath,
							extList: statusRef.current ? (statusRef.current.extensions || []) : [],
						}),
				}),
				msg ? jsx("div", { style: { position: "fixed", bottom: 18, right: 18, maxWidth: 420, zIndex: 90 }, children: jsx("div", { className: "pm-msg " + msg.kind, onClick: function () { setMsg(null); }, children: msg.text }) }) : null,
				wakeToast ? jsx("div", { className: "pm-wake", children: jsxs("span", { children: [jsx("b", { children: "☀ 它醒来了" }), "（第 " + wakeToast.count + " 次苏醒）"] }) }) : null,
			] });
		}

		var inject = ["slots"];

		function apply(ctx) {
			ensureCss();
			ctx.slots.inject("sidebar.panellist", function () {
				return ctx.slots.register({
					name: "sidebar.panellist",
					id: PANEL_ID,
					order: 210,
					label: function () { return TITLE; },
				}, Icon);
			});
			ctx.slots.inject("main", function () {
				return ctx.slots.register({
					name: "main",
					key: PANEL_ID,
				}, function () { return Root(ctx); });
			});
		}

		exports.name = "dsh-plastic-memory";
		exports.inject = inject;
		exports.apply = apply;
		return module.exports;
	},
});
