import React, { useState } from "react";
import { useAuth } from "./contexts/AuthContext";
import { AuthPage } from "./components/auth/AuthPage";
import { ChatPage } from "./components/chat/ChatPage";
import { ProfilePage } from "./components/profile/ProfilePage";
import { FeedbackPage } from "./components/feedback/FeedbackPage";

type Page = "chat" | "profile" | "feedback";

export default function App() {
  const { user } = useAuth();
  const [page, setPage] = useState<Page>("chat");

  if (!user) return <AuthPage />;
  if (page === "profile") return <ProfilePage onBack={() => setPage("chat")} />;
  if (page === "feedback") return <FeedbackPage onBack={() => setPage("chat")} />;
  return <ChatPage onOpenProfile={() => setPage("profile")} onOpenFeedback={() => setPage("feedback")} />;
}
