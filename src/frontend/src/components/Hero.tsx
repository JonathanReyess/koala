import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom"; // Vite standard for routing

interface HeroProps {
  onStartLearning: () => void;
}

export const Hero = ({ onStartLearning }: HeroProps) => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col relative bg-white dark:bg-black">
      {/* Header with backdrop blur - Apple style */}
      <header className="w-full flex items-center justify-between px-6 md:px-12 py-4 absolute top-0 left-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-black/80 border-b border-gray-200/20 dark:border-gray-800/20">
        <img
          src="/koala_logo.svg"
          alt="Koala Logo"
          onClick={() => navigate("/")}
          className="h-16 md:h- w-auto cursor-pointer hover:opacity-80 transition-opacity mix-blend-multiply dark:mix-blend-screen"
        />

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            onClick={() => navigate("/login")}
            className="text-sm md:text-base font-medium px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-900 rounded-full transition-colors"
          >
            Log in
          </Button>
          <Button
            variant="default"
            onClick={() => navigate("/signup")}
            className="text-sm md:text-base font-medium px-5 py-2 rounded-full transition-all hover:opacity-90"
          >
            Sign up
          </Button>
        </div>
      </header>

      {/* Hero content */}
      <main className="flex flex-1 items-center justify-center px-6 pt-32 pb-16">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-semibold tracking-tight leading-none mb-6">
            Learn Korean
            <br />
            Sign Language
          </h1>

          <p className="text-xl md:text-2xl lg:text-3xl text-gray-600 dark:text-gray-400 font-normal mb-12 max-w-3xl mx-auto leading-snug">
            Master Korean Sign Language through interactive practice with
            real-time feedback.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-20">
            <Button
              size="lg"
              onClick={onStartLearning}
              className="w-full sm:w-auto text-base md:text-lg font-medium px-8 py-6 rounded-full transition-all hover:opacity-90"
            >
              Practice
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>

            <a
              href="https://sldict.korean.go.kr/front/main/main.do"
              target="_blank"
              rel="noopener noreferrer"
              className="w-full sm:w-auto"
            >
              <Button
                size="lg"
                variant="outline"
                className="w-full sm:w-auto text-base md:text-lg font-medium px-8 py-6 rounded-full border-2 hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
              >
                Dictionary
              </Button>
            </a>
          </div>

          {/* Features */}
          <div className="grid grid-cols-2 gap-8 md:gap-16 max-w-2xl mx-auto pt-8 border-t border-gray-200/50 dark:border-gray-800/50">
            <div className="space-y-2">
              <div className="text-5xl md:text-6xl font-semibold text-primary tracking-tight">
                100%
              </div>
              <div className="text-base md:text-lg text-gray-600 dark:text-gray-400 font-normal">
                Free Forever
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-5xl md:text-6xl font-semibold text-primary tracking-tight">
                AI
              </div>
              <div className="text-base md:text-lg text-gray-600 dark:text-gray-400 font-normal">
                Powered Learning
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
