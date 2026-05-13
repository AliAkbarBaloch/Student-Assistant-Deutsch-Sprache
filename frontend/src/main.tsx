import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { AuthProvider } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { LevelProvider } from "./contexts/LevelContext";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <LevelProvider>
          <App />
        </LevelProvider>
      </AuthProvider>
    </ThemeProvider>
  </React.StrictMode>
);
