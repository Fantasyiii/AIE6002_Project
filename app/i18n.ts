export type Lang = "en" | "zh";

export const translations = {
  en: {
    placeholder: "Describe the vibe you're looking for...",
    suggestions: 'Try: "sci-fi about time travel" or "romantic comedy for weekend"',
    thinking: "Thinking...",
    sources: "Sources",
    responseTime: "Response time:",
    errorMsg:
      "Sorry, I encountered an error. Please make sure the backend server is running.",
    langToggle: "中文",
  },
  zh: {
    placeholder: "描述你想要的电影氛围...",
    suggestions: '试试："关于时间旅行的科幻片" 或 "周末看的爱情喜剧"',
    thinking: "思考中...",
    sources: "推荐来源",
    responseTime: "响应时间：",
    errorMsg: "抱歉，搜索时发生错误。请确保后端服务正在运行。",
    langToggle: "English",
  },
} as const;

export type Translations = (typeof translations)[Lang];
