import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Learn from "./pages/Learn";
import { lazy, Suspense } from "react";

// Dev-only landmark parity tool; tree-shaken out of production builds.
const DebugParity = import.meta.env.DEV ? lazy(() => import("./pages/DebugParity")) : null;
// New Imports for Auth Pages
import { LoginPage } from "./pages/LoginPage";
import { SignupPage } from "./pages/SignupPage";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
      <Routes>
      <Route path="/" element={<Index />} />
      <Route path="/learn" element={<Learn />} />
      {DebugParity && (
        <Route
          path="/debug/parity"
          element={
            <Suspense fallback={null}>
              <DebugParity />
            </Suspense>
          }
        />
      )}
      {/* New Routes for Authentication */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;