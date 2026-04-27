import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ArrowRight, Globe } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";

const JumpyText = ({
  text,
  indexOffset = 0,
}: {
  text: string;
  indexOffset?: number;
}) => {
  const shouldReduceMotion = useReducedMotion();
  return (
    <>
      {text.split("").map((char, i) =>
        char === " " ? (
          <span key={i} className="inline-block">
            &nbsp;
          </span>
        ) : (
          <motion.span
            key={i}
            className="inline-block cursor-default"
            animate={shouldReduceMotion ? {} : { y: [0, -12, 0] }}
            transition={{
              delay: (indexOffset + i) * 0.04,
              duration: 0.5,
              ease: [0.22, 1, 0.36, 1],
            }}
            whileHover={
              shouldReduceMotion
                ? {}
                : {
                    y: -10,
                    transition: { type: "spring", stiffness: 500, damping: 12 },
                  }
            }
          >
            {char}
          </motion.span>
        ),
      )}
    </>
  );
};

interface HeroProps {
  onStartLearning: () => void;
  onOpenDictionary?: () => void;
}

export const Hero = ({ onStartLearning, onOpenDictionary }: HeroProps) => {
  const navigate = useNavigate();
  const shouldReduceMotion = useReducedMotion();
  const [isKorean, setIsKorean] = useState(false);
  const [isWiggling, setIsWiggling] = useState(false);

  const textVariants = shouldReduceMotion
    ? { initial: {}, animate: {}, exit: {} }
    : {
        initial: { opacity: 0, y: 5 },
        animate: { opacity: 1, y: 0 },
        exit: { opacity: 0, y: -5 },
      };

  const transitionConfig = shouldReduceMotion
    ? { duration: 0 }
    : { duration: 0.6, ease: [0.4, 0, 0.2, 1] as const };

  const toggleLanguage = () => setIsKorean((prev) => !prev);

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-white dark:bg-black">
      {/* Header */}
      <header className="w-full flex items-center justify-between px-6 md:px-12 py-4 absolute top-0 left-0 z-50 bg-transparent">
        <img
          src="/koala_logo.png"
          alt="Koala - Korean Sign Language Learning"
          onClick={() => navigate("/")}
          onKeyDown={(e) => e.key === "Enter" && navigate("/")}
          tabIndex={0}
          role="button"
          className="h-20 w-auto cursor-pointer hover:opacity-80 transition-opacity mix-blend-multiply dark:mix-blend-screen focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
        />

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            onClick={toggleLanguage}
            className="rounded-full flex items-center gap-2"
            aria-label={isKorean ? "Switch to English" : "Switch to Korean"}
          >
            <Globe className="size-7" />
            <span className="text-base font-semibold tracking-wide">
              {isKorean ? "KOR" : "ENG"}
            </span>
          </Button>

          {/* <Button
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
          </Button> */}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex flex-1 items-center justify-center px-6 pt-32 pb-16">
        <div className="max-w-5xl mx-auto text-center">
          {/* Animated Heading */}
          <h1
            className="text-5xl md:text-7xl lg:text-8xl font-semibold tracking-tight leading-[1.1] mb-6 relative"
            lang={isKorean ? "ko" : "en"}
          >
            <span className="invisible block" aria-hidden="true">
              Learn Korean
              <br />
              Sign Language
            </span>

            <AnimatePresence mode="wait">
              <motion.span
                key={isKorean ? "kr" : "en"}
                variants={textVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="absolute inset-0 flex flex-col items-center justify-center"
                transition={transitionConfig}
                aria-live="polite"
              >
                {isKorean ? (
                  <>
                    <span>
                      <JumpyText text="한국 수어" indexOffset={0} />
                    </span>
                    <span>
                      <JumpyText
                        text="배우기"
                        indexOffset={"한국 수어".length}
                      />
                    </span>
                  </>
                ) : (
                  <>
                    <span>
                      <JumpyText text="Learn Korean" indexOffset={0} />
                    </span>
                    <span>
                      <JumpyText
                        text="Sign Language"
                        indexOffset={"Learn Korean".length}
                      />
                    </span>
                  </>
                )}
              </motion.span>
            </AnimatePresence>
          </h1>

          {/* Animated Subtext */}
          <div
            className="text-xl md:text-2xl lg:text-3xl text-gray-600 dark:text-gray-400 font-normal mb-12 max-w-3xl mx-auto leading-relaxed relative"
            lang={isKorean ? "ko" : "en"}
          >
            <span className="invisible block" aria-hidden="true">
              Master Korean Sign Language through interactive practice with
              real-time feedback.
            </span>

            <AnimatePresence mode="wait">
              <motion.p
                key={isKorean ? "kr-p" : "en-p"}
                variants={textVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="absolute inset-0 flex items-center justify-center"
                transition={{
                  ...transitionConfig,
                  delay: shouldReduceMotion ? 0 : 0.05,
                }}
                aria-live="polite"
              >
                {isKorean
                  ? "실시간 피드백과 함께 상호작용하며 한국 수어를 마스터하세요."
                  : "Master Korean Sign Language through interactive practice with real-time feedback."}
              </motion.p>
            </AnimatePresence>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-20">
            <motion.div
              animate={
                isWiggling && !shouldReduceMotion
                  ? { rotate: [0, -6, 6, 0] }
                  : { rotate: 0 }
              }
              transition={{ duration: 0.5, ease: "easeInOut" }}
              onHoverStart={() => {
                if (!shouldReduceMotion) setIsWiggling(true);
              }}
              onAnimationComplete={() => setIsWiggling(false)}
            >
              <Button
                size="lg"
                onClick={onStartLearning}
                className="w-full sm:w-auto px-12 py-7 rounded-full text-lg"
              >
                {isKorean ? "연습하기" : "Practice"}
                <ArrowRight className="ml-2 h-5 w-5" aria-hidden="true" />
              </Button>
            </motion.div>
          </div>

          {/* Features Section */}
          <div className="grid grid-cols-2 gap-8 md:gap-16 max-w-2xl mx-auto pt-8 border-t border-gray-200/50 dark:border-gray-800/50">
            <div className="space-y-2">
              <div className="text-5xl md:text-6xl font-semibold text-primary tracking-tight">
                100%
              </div>
              <div
                className="text-base md:text-lg text-gray-600 dark:text-gray-400"
                lang={isKorean ? "ko" : "en"}
              >
                {isKorean ? "영원히 무료" : "Free Forever"}
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-5xl md:text-6xl font-semibold text-primary tracking-tight">
                AI
              </div>
              <div
                className="text-base md:text-lg text-gray-600 dark:text-gray-400"
                lang={isKorean ? "ko" : "en"}
              >
                {isKorean ? "인공지능 기반 학습" : "AI Powered Learning"}
              </div>
            </div>
          </div>
        </div>
      </main>
      {/* Peeking koala */}
      <img
        src="/koala_wave.png"
        alt=""
        aria-hidden="true"
        className="absolute bottom-2 right-12 w-48 md:w-72 translate-y-1/4 pointer-events-none select-none"
      />
    </div>
  );
};
