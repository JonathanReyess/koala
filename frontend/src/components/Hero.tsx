import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { useIsMobile } from "@/hooks/use-mobile";
import { useNavigate } from "react-router-dom";

interface HeroProps {
  onStartLearning: () => void;
}

export const Hero = ({ onStartLearning }: HeroProps) => {
  const isMobile = useIsMobile();
  // Initialize the navigate function
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col relative">
      {/* Header with responsive logo and auth buttons */}
      <header className="w-full flex items-center justify-between p-6 absolute top-0 left-0 z-50">
        {/* Logo (Left side) */}
        <img
          src="/koala_logo.svg"
          alt="Koala Logo"
          style={{
            width: isMobile ? "172px" : "250px",
            height: "auto",
          }}
          className="cursor-pointer hover:opacity-90 transition-opacity"
        />

{/* Log-in and Sign-up Buttons (Right side) */}
<div className="flex items-center space-x-4 mr-2 -mt-4">
          <Button
            variant="ghost"
            className="text-lg hover:bg-primary/10 transition-all duration-300"
            // Use navigate to go to the /login path
            onClick={() => navigate("/login")} 
          >
            Log-in
          </Button>
          <Button
            variant="default"
            className="text-lg px-6 py-2 transition-all duration-300 hover:scale-105"
            // Use navigate to go to the /signup path
            onClick={() => navigate("/signup")} 
          >
            Sign-up
          </Button>
        </div>
      </header>

{/* Hero content */}
<main className="flex flex-1 items-center justify-center px-4">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h1 className="text-6xl md:text-7xl font-bold tracking-tight">
              Learn Korean<br />Sign Language
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
            Master Korean Sign Language through <br />interactive practice with real-time feedback.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            {/* Learn (secondary) */}
            <a
              href="https://sldict.korean.go.kr/front/main/main.do"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button
                size="lg"
                variant="outline"
                className="text-lg px-8 mt-4 py-6 border-primary text-primary hover:bg-primary/10 transition-all duration-300 hover:scale-105"
              >
                Dictionary
              </Button>
            </a>

            {/* Practice (primary) */}
            <a href="/learn">
              <Button
                size="lg"
                className="text-lg px-8 py-6 mt-4 bg-primary hover:bg-primary/90 transition-all duration-300 hover:scale-105"
              >
                Practice
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </a>
          </div>

          {/* Features (without spaced repetition) */}
          <div className="pt-6 grid grid-cols-1 md:grid-cols-2 gap-8 max-w-2xl mx-auto">
            <div className="space-y-2">
            <div className="text-4xl md:text-5xl font-bold text-primary">100%</div>

            <div className="text-base md:text-lg text-muted-foreground">Free Forever</div>

            </div>
            <div className="space-y-2">
            <div className="text-4xl md:text-5xl font-bold text-primary">AI</div>
              <div className="text-base md:text-lg text-muted-foreground">Powered Learning</div>

            </div>
          </div>
        </div>
      </main>
    </div>
  );
};