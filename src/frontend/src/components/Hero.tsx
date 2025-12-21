import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";

interface HeroProps {
  onStartLearning: () => void;
}

export const Hero = ({ onStartLearning }: HeroProps) => {
  const navigate = useNavigate();
  const [isKorean, setIsKorean] = useState(false);

  // Logic to toggle language every 7 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setIsKorean((prev) => !prev);
    }, 6500);
    return () => clearInterval(interval);
  }, []);

  // Animation variants with smoother transitions
  const textVariants = {
    initial: { opacity: 0, y: 5 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -5 },
  };

  return (
    <div className="min-h-screen flex flex-col relative bg-white dark:bg-black">
      <header className="w-full flex items-center justify-between px-6 md:px-12 py-4 absolute top-0 left-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-black/80 border-b border-gray-200/20 dark:border-gray-800/20">
        <img
          src="/koala_logo.svg"
          alt="Koala Logo"
          onClick={() => navigate("/")}
          className="h-16 w-auto cursor-pointer hover:opacity-80 transition-opacity mix-blend-multiply dark:mix-blend-screen"
        />

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            onClick={() => navigate("/login")}
            className="rounded-full"
          >
            {isKorean ? "로그인" : "Log in"}
          </Button>
          <Button
            variant="default"
            onClick={() => navigate("/signup")}
            className="rounded-full"
          >
            {isKorean ? "가입하기" : "Sign up"}
          </Button>
        </div>
      </header>

      <main className="flex flex-1 items-center justify-center px-6 pt-32 pb-16">
        <div className="max-w-5xl mx-auto text-center">
          {/* --- Animated Heading --- */}
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-semibold tracking-tight leading-[1.1] mb-6 relative">
            {/* Ghost: Ensures the height never changes */}
            <div className="invisible" aria-hidden="true">
              Learn Korean
              <br />
              Sign Language
            </div>

            <AnimatePresence mode="wait">
              <motion.span
                key={isKorean ? "kr" : "en"}
                variants={textVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="absolute inset-0 flex items-center justify-center"
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
              >
                {isKorean ? (
                  <>
                    한국 수어
                    <br />
                    배우기
                  </>
                ) : (
                  <>
                    Learn Korean
                    <br />
                    Sign Language
                  </>
                )}
              </motion.span>
            </AnimatePresence>
          </h1>

          {/* --- Animated Subtext --- */}
          <div className="text-xl md:text-2xl lg:text-3xl text-gray-600 dark:text-gray-400 font-normal mb-12 max-w-3xl mx-auto leading-relaxed relative">
            {/* Ghost: Keeps the paragraph container height stable */}
            <div className="invisible" aria-hidden="true">
              Master Korean Sign Language through interactive practice with
              real-time feedback.
            </div>

            <AnimatePresence mode="wait">
              <motion.p
                key={isKorean ? "kr-p" : "en-p"}
                variants={textVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="absolute inset-0 flex items-center justify-center"
                transition={{
                  duration: 0.6,
                  ease: [0.4, 0, 0.2, 1],
                  delay: 0.05,
                }}
              >
                {isKorean
                  ? "실시간 피드백과 함께 상호작용하며 한국 수어를 마스터하세요."
                  : "Master Korean Sign Language through interactive practice with real-time feedback."}
              </motion.p>
            </AnimatePresence>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-20">
            <Button
              size="lg"
              onClick={onStartLearning}
              className="w-full sm:w-auto px-8 py-6 rounded-full"
            >
              {isKorean ? "연습하기" : "Practice"}
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="w-full sm:w-auto px-8 py-6 rounded-full border-2"
            >
              {isKorean ? "사전" : "Dictionary"}
            </Button>
          </div>

          {/* Features */}
          <div className="grid grid-cols-2 gap-8 md:gap-16 max-w-2xl mx-auto pt-8 border-t border-gray-200/50 dark:border-gray-800/50">
            <div className="space-y-2">
              <div className="text-5xl md:text-6xl font-semibold text-primary tracking-tight">
                100%
              </div>
              <div className="text-base md:text-lg text-gray-600 dark:text-gray-400">
                {isKorean ? "영원히 무료" : "Free Forever"}
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-5xl md:text-6xl font-semibold text-primary tracking-tight">
                AI
              </div>
              <div className="text-base md:text-lg text-gray-600 dark:text-gray-400">
                {isKorean ? "인공지능 기반 학습" : "AI Powered Learning"}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
