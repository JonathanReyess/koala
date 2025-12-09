import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
// Assuming useIsMobile is working correctly and provided by the user
import { useIsMobile } from "@/hooks/use-mobile";
import { useNavigate } from "react-router-dom";

interface HeroProps {
  onStartLearning: () => void;
}

export const Hero = ({ onStartLearning }: HeroProps) => {
  const isMobile = useIsMobile();
  const navigate = useNavigate();

  return (
    // Ensure the container fills the screen and is ready for responsive stacking
    <div className="min-h-screen flex flex-col relative ">
      {/* Header: Use 'p-4 md:p-6' for less padding on mobile, and ensure button alignment */}
      <header className="w-full flex items-center justify-between p-4 md:p-6 absolute top-0 left-0 z-50">
        {/* Logo (Left side) - Simplified width management */}
        <img
          src="/koala_logo.svg"
          alt="Koala Logo"
          // Removed inline style and used Tailwind classes for width control
          className="h-auto w-[120px] sm:w-[172px] md:w-[250px] cursor-pointer hover:opacity-90 transition-opacity"
        />

        {/* Log-in and Sign-up Buttons (Right side) - Adjusted for better mobile alignment */}
        <div className="flex items-center space-x-2 sm:space-x-4">
          <Button
            variant="ghost"
            // Reduced font size for mobile buttons and ensured smaller padding
            className="text-sm sm:text-lg px-2 sm:px-4 py-1 sm:py-2 hover:bg-primary/10 transition-all duration-300 h-auto"
            onClick={() => navigate("/login")}
          >
            Log-in
          </Button>
          <Button
            variant="default"
            // Reduced font size and padding for mobile
            className="text-sm sm:text-lg px-3 sm:px-6 py-1.5 sm:py-2 transition-all duration-300 hover:scale-105 h-auto"
            onClick={() => navigate("/signup")}
          >
            Sign-up
          </Button>
        </div>
      </header>

      {/* Hero content */}
      {/* Increased top padding for mobile to clear the fixed header */}
      <main className="flex flex-1 items-center justify-center pt-28 pb-12 sm:py-16 px-4">
        <div className="max-w-4xl mx-auto text-center space-y-6 sm:space-y-8">
          <div className="space-y-3 sm:space-y-4">
            <h1 className="text-4xl sm:text-6xl md:text-7xl font-bold tracking-tight leading-tight">
              Learn Korean<br />Sign Language
            </h1>
            <p className="text-base sm:text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
              Master Korean Sign Language through <br />interactive practice with real-time feedback.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            {/* Dictionary (secondary) */}
            <a
              href="https://sldict.korean.go.kr/front/main/main.do"
              target="_blank"
              rel="noopener noreferrer"
              className="w-full sm:w-auto" // Added w-full for full-width on mobile
            >
              <Button
                size="lg"
                variant="outline"
                className="w-full sm:w-auto text-lg px-8 py-6 border-primary text-primary hover:bg-primary/10 transition-all duration-300 hover:scale-105"
              >
                Dictionary
              </Button>
            </a>

            {/* Practice (primary) */}
            <a href="/learn" className="w-full sm:w-auto"> {/* Added w-full for full-width on mobile */}
              <Button
                size="lg"
                className="w-full sm:w-auto text-lg px-8 py-6 bg-primary hover:bg-primary/90 transition-all duration-300 hover:scale-105"
              >
                Practice
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </a>
          </div>

          {/* Features */}
          <div className="pt-6 grid grid-cols-2 gap-4 sm:gap-8 max-w-2xl mx-auto">
            <div className="space-y-1 sm:space-y-2">
              {/* Scaled down font size for mobile */}
              <div className="text-3xl sm:text-4xl md:text-5xl font-bold text-primary">100%</div>
              <div className="text-sm sm:text-base md:text-lg text-muted-foreground">Free Forever</div>
            </div>
            <div className="space-y-1 sm:space-y-2">
              {/* Scaled down font size for mobile */}
              <div className="text-3xl sm:text-4xl md:text-5xl font-bold text-primary">AI</div>
              <div className="text-sm sm:text-base md:text-lg text-muted-foreground">Powered Learning</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};