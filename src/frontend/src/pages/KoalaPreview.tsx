import { useState } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { Hero } from "@/components/Hero";
import { KOALA_VARIANTS, type KoalaVariant } from "@/components/KoalaIntro";

// Dev-only: compare the koala intro shrink variants at /koala/<variant>.
const KoalaPreview = () => {
  const { variant } = useParams();
  const navigate = useNavigate();
  const [replay, setReplay] = useState(0);

  if (!KOALA_VARIANTS.includes(variant as KoalaVariant)) {
    return <Navigate to={`/koala/${KOALA_VARIANTS[0]}`} replace />;
  }

  return (
    <>
      <Hero
        key={`${variant}-${replay}`}
        onStartLearning={() => navigate("/learn")}
        koalaVariant={variant as KoalaVariant}
        alwaysPlayKoalaIntro
      />
      <nav className="fixed top-24 left-1/2 -translate-x-1/2 z-[70] flex flex-wrap items-center gap-1 rounded-full border bg-white/90 dark:bg-black/90 px-2 py-1 text-sm shadow">
        {KOALA_VARIANTS.map((v) => (
          <Link
            key={v}
            to={`/koala/${v}`}
            className={`rounded-full px-3 py-1 ${
              v === variant
                ? "bg-primary text-primary-foreground"
                : "hover:bg-gray-100 dark:hover:bg-gray-800"
            }`}
          >
            {v}
          </Link>
        ))}
        <button
          type="button"
          onClick={() => setReplay((n) => n + 1)}
          className="rounded-full px-3 py-1 font-semibold hover:bg-gray-100 dark:hover:bg-gray-800"
        >
          ↻ Replay
        </button>
      </nav>
    </>
  );
};

export default KoalaPreview;
