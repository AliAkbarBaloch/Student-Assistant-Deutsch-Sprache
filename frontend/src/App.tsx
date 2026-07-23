import React, { useState } from "react";
import { useAuth } from "./contexts/AuthContext";
import { AuthPage } from "./components/auth/AuthPage";
import { ChatPage } from "./components/chat/ChatPage";
import { ProfilePage } from "./components/profile/ProfilePage";
import { FeedbackPage } from "./components/feedback/FeedbackPage";
import { HistoryPage } from "./components/history/HistoryPage";

type Page = "chat" | "profile" | "feedback" | "history";

export default function App() {
  const { user } = useAuth();
  const [page, setPage] = useState<Page>("chat");

  if (!user) return <AuthPage />;
  if (page === "profile") return <ProfilePage onBack={() => setPage("chat")} onOpenHistory={() => setPage("history")} />;
  if (page === "feedback") return <FeedbackPage onBack={() => setPage("chat")} />;
  if (page === "history") return <HistoryPage onBack={() => setPage("chat")} />;
  return (
    <ChatPage
      onOpenProfile={() => setPage("profile")}
      onOpenFeedback={() => setPage("feedback")}
      onOpenHistory={() => setPage("history")}
    />
  );
}
