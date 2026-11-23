import express from "express";
import type { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PRIMARY_MODEL = "gemini-4-hydrogen";
const FALLBACK_MODEL = "gemini-3.1-flash-lite";

interface QuestionItem {
  points: number;
  type: "standard";
  question: string;
  answer: string;
  rationale: string;
}

interface CategoryItem {
  category: string;
  questions: QuestionItem[];
}

interface BonusQuestionItem {
  category: string;
  points: number;
  question: string;
  answer: string;
  rationale: string;
}

interface FinalJeopardyItem {
  category: string;
  question: string;
  answer: string;
  rationale: string;
}

interface QuizDataPayload {
  title: string;
  topic: string;
  round1: CategoryItem[];
  bonus1: BonusQuestionItem;
  round2: CategoryItem[];
  bonus2: BonusQuestionItem;
  finalJeopardy: FinalJeopardyItem;
}

let cachedClient: GoogleGenAI | null = null;
let cachedKey: string | undefined;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("API_KEY_MISSING: GEMINI_API_KEY nu este configurată pe server.");
  }
  if (!cachedClient || cachedKey !== apiKey) {
    cachedClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
    cachedKey = apiKey;
  }
  return cachedClient;
}

function truncateText(text: string, maxLength: number = 15000): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "... [conținut trunchiat]";
}

function extractJSON(text: string): string {
  let cleaned = text.trim();
  cleaned = cleaned
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/, "")
    .replace(/\s*```$/, "");

  const firstBrace = cleaned.indexOf("{");
  const lastBrace = cleaned.lastIndexOf("}");

  if (firstBrace === -1 || lastBrace === -1) {
    throw new Error("Nu s-a găsit niciun obiect JSON în răspuns");
  }

  return cleaned.substring(firstBrace, lastBrace + 1);
}

function tryFixJSON(jsonStr: string): string {
  let fixed = jsonStr.trim();

  fixed = fixed.replace(/,\s*$/g, "");
  fixed = fixed.replace(/,\s*}/g, "}");
  fixed = fixed.replace(/,\s*]/g, "]");

  const openBraces = (fixed.match(/\{/g) || []).length;
  const closeBraces = (fixed.match(/\}/g) || []).length;
  const openBrackets = (fixed.match(/\[/g) || []).length;
  const closeBrackets = (fixed.match(/\]/g) || []).length;

  const missingBrackets = openBrackets - closeBrackets;
  const missingBraces = openBraces - closeBraces;

  for (let i = 0; i < missingBrackets; i++) fixed += "]";
  for (let i = 0; i < missingBraces; i++) fixed += "}";

  return fixed;
}

function buildPrompt(
  topic: string,
  difficulty: "easy" | "medium" | "hard",
  rounds: number,
  sourceContent?: string
): string {
  const difficultyMap = {
    easy: "ușor, pentru începători",
    medium: "mediu, pentru nivel intermediar",
    hard: "avansat, pentru experți",
  };

  const hasSourceContent = Boolean(sourceContent && sourceContent.trim().length > 0);

  const sourceSection = hasSourceContent
    ? `\n\n=== CONȚINUT SURSĂ (generează întrebările EXCLUSIV pe baza acestui conținut) ===\n${truncateText(
        sourceContent!
      )}\n=== SFÂRȘIT CONȚINUT SURSĂ ===\n\nIMPORTANT: Toate întrebările, răspunsurile și explicațiile trebuie să fie bazate STRICT pe conținutul de mai sus. Nu inventa informații care nu apar în textul sursă.`
    : "";

  return `Generează un concurs tip Jeopardy/Trivia pe tema "${topic}" cu dificultate ${difficultyMap[difficulty]}.${sourceSection}

Creează ${rounds} rund${rounds === 1 ? "ă" : "e"} principal${rounds === 1 ? "ă" : "e"}, fiecare cu 5 categorii și 4 întrebări per categorie.
Pentru fiecare rundă, creează și o întrebare bonus.
La final, creează o întrebare pentru Final Jeopardy.

Reguli stricte:
- Răspunde DOAR cu JSON valid, conform schemei cerute
- Fiecare categorie trebuie să aibă EXACT 4 întrebări
- Punctajele pentru runda 1: 100, 200, 300, 400
- Punctajele pentru runda 2: 200, 400, 800, 1000
- Toate întrebările, răspunsurile, categoriile și explicațiile trebuie să fie în limba română
- Întrebările trebuie să fie variate și interesante, specifice temei "${topic}"
- Răspunsurile trebuie să fie concise și corecte
- Explicațiile (rationale) trebuie să fie scurte dar informative
${hasSourceContent ? "- Bazează-te EXCLUSIV pe informațiile din conținutul sursă furnizat" : ""}
${rounds === 1 ? "NOTĂ: Dacă este doar 1 rundă, completează round2 cu aceleași categorii dar cu întrebări diferite." : ""}

Generează ACUM JSON-ul complet pentru tema "${topic}".`;
}

const questionSchema = {
  type: Type.OBJECT,
  properties: {
    points: { type: Type.INTEGER },
    type: { type: Type.STRING },
    question: { type: Type.STRING },
    answer: { type: Type.STRING },
    rationale: { type: Type.STRING },
  },
  required: ["points", "type", "question", "answer", "rationale"],
};

const categorySchema = {
  type: Type.OBJECT,
  properties: {
    category: { type: Type.STRING },
    questions: {
      type: Type.ARRAY,
      items: questionSchema,
    },
  },
  required: ["category", "questions"],
};

const bonusQuestionSchema = {
  type: Type.OBJECT,
  properties: {
    category: { type: Type.STRING },
    points: { type: Type.INTEGER },
    question: { type: Type.STRING },
    answer: { type: Type.STRING },
    rationale: { type: Type.STRING },
  },
  required: ["category", "points", "question", "answer", "rationale"],
};

const quizResponseSchema = {
  type: Type.OBJECT,
  properties: {
    title: { type: Type.STRING },
    topic: { type: Type.STRING },
    round1: {
      type: Type.ARRAY,
      items: categorySchema,
    },
    bonus1: bonusQuestionSchema,
    round2: {
      type: Type.ARRAY,
      items: categorySchema,
    },
    bonus2: bonusQuestionSchema,
    finalJeopardy: {
      type: Type.OBJECT,
      properties: {
        category: { type: Type.STRING },
        question: { type: Type.STRING },
        answer: { type: Type.STRING },
        rationale: { type: Type.STRING },
      },
      required: ["category", "question", "answer", "rationale"],
    },
  },
  required: ["title", "topic", "round1", "bonus1", "round2", "bonus2", "finalJeopardy"],
};

function normalizeQuizData(raw: Record<string, unknown>, topic: string): QuizDataPayload | null {
  const quizData = { ...raw } as Partial<QuizDataPayload>;

  if (!quizData.title) quizData.title = `Concurs: ${topic}`;
  if (!quizData.topic) quizData.topic = topic;

  if (!quizData.round1 || !Array.isArray(quizData.round1) || quizData.round1.length === 0) {
    return null;
  }

  if (!quizData.round2 || !Array.isArray(quizData.round2) || quizData.round2.length === 0) {
    quizData.round2 = quizData.round1;
  }

  if (!quizData.bonus1) {
    quizData.bonus1 = {
      category: "Bonus Runda 1",
      points: 300,
      question: "Întrebare bonus?",
      answer: "Răspuns",
      rationale: "Explicație",
    };
  }

  if (!quizData.bonus2) {
    quizData.bonus2 = {
      category: "Bonus Runda 2",
      points: 500,
      question: "Întrebare bonus?",
      answer: "Răspuns",
      rationale: "Explicație",
    };
  }

  if (!quizData.finalJeopardy) {
    return null;
  }

  const ensureQuestions = (categories: CategoryItem[], roundNumber: 1 | 2): CategoryItem[] => {
    const defaultPoints = roundNumber === 1 ? [100, 200, 300, 400] : [200, 400, 800, 1000];
    return categories.map((cat) => ({
      category: cat.category || "Categorie",
      questions: Array.isArray(cat.questions)
        ? cat.questions.map((q, i) => ({
            points: typeof q.points === "number" && q.points > 0 ? q.points : defaultPoints[i] ?? (i + 1) * 100,
            type: "standard" as const,
            question: q.question || "Întrebare?",
            answer: q.answer || "Răspuns",
            rationale: q.rationale || "Explicație",
          }))
        : [],
    }));
  };

  quizData.round1 = ensureQuestions(quizData.round1, 1);
  quizData.round2 = ensureQuestions(quizData.round2, 2);

  return quizData as QuizDataPayload;
}

function getHttpStatusFromError(error: unknown): number {
  const err = error as { status?: number; code?: number; message?: string } | null;
  if (typeof err?.status === "number" && err.status >= 400 && err.status < 600) {
    return err.status;
  }
  const msg = err?.message || "";
  if (msg.includes("API_KEY_INVALID") || msg.includes("401")) return 401;
  if (msg.includes("PERMISSION_DENIED") || msg.includes("403")) return 403;
  if (msg.includes("RESOURCE_EXHAUSTED") || msg.includes("429") || msg.includes("quota")) return 429;
  if (msg.includes("NOT_FOUND") || msg.includes("404")) return 404;
  return 500;
}

async function generateQuizServerSide(
  topic: string,
  difficulty: "easy" | "medium" | "hard",
  rounds: number,
  sourceContent?: string
): Promise<QuizDataPayload> {
  const ai = getGeminiClient();
  const prompt = buildPrompt(topic, difficulty, rounds, sourceContent);
  const models = [PRIMARY_MODEL, FALLBACK_MODEL];
  const maxRetries = 2;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    for (const model of models) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: prompt,
          config: {
            systemInstruction:
              "Ești un creator expert de concursuri tip Jeopardy în limba română. Răspunzi strict cu JSON valid care respectă schema cerută, concis și precis.",
            responseMimeType: "application/json",
            responseSchema: quizResponseSchema,
            ...(model === PRIMARY_MODEL
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : {}),
          },
        });

        const textResponse = response.text;
        if (!textResponse) {
          throw new Error("Nu s-a primit răspuns de la AI");
        }

        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(textResponse) as Record<string, unknown>;
        } catch {
          const jsonText = extractJSON(textResponse);
          try {
            parsed = JSON.parse(jsonText) as Record<string, unknown>;
          } catch {
            const fixed = tryFixJSON(jsonText);
            parsed = JSON.parse(fixed) as Record<string, unknown>;
          }
        }

        const normalized = normalizeQuizData(parsed, topic);
        if (!normalized) {
          throw new Error("JSON incomplet generat de AI");
        }

        return normalized;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        const status = getHttpStatusFromError(error);

        // Nu reîncerca pentru erori de autentificare sau permisiuni
        if (status === 401 || status === 403 || lastError.message.includes("API_KEY_MISSING")) {
          throw lastError;
        }

        // Pentru erori de JSON sau model indisponibil, încearcă modelul următor
        if (
          status === 404 ||
          lastError.message.includes("JSON") ||
          lastError.message.includes("not found") ||
          lastError.message.includes("not available") ||
          lastError.message.includes("not supported")
        ) {
          continue;
        }

        // Backoff scurt pentru 429 sau erori tranzitorii
        if (status === 429 || status >= 500) {
          await new Promise((resolve) => setTimeout(resolve, 600 * (attempt + 1)));
          continue;
        }

        throw lastError;
      }
    }
  }

  throw lastError || new Error("Eroare necunoscută la generarea concursului");
}

async function startServer() {
  const app = express();
  const port = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: "5mb" }));

  app.post("/api/gemini/generate-quiz", async (req: Request, res: Response) => {
    try {
      const { topic, difficulty, rounds, sourceContent } = req.body ?? {};

      if (typeof topic !== "string" || !topic.trim()) {
        res.status(400).json({ error: "Te rog introdu un subiect!" });
        return;
      }

      const validDifficulty: "easy" | "medium" | "hard" =
        difficulty === "easy" || difficulty === "medium" || difficulty === "hard"
          ? difficulty
          : "medium";

      const validRounds = rounds === 1 ? 1 : 2;
      const validSource = typeof sourceContent === "string" ? sourceContent : undefined;

      const quizData = await generateQuizServerSide(
        topic.trim().slice(0, 500),
        validDifficulty,
        validRounds,
        validSource
      );

      res.json(quizData);
    } catch (error) {
      console.error("Failed to generate quiz via Gemini API:", error);
      const status = getHttpStatusFromError(error);
      const message =
        error instanceof Error ? error.message : "Eroare la generarea concursului";
      res.status(status).json({ error: message });
    }
  });

  const distDir = path.resolve(__dirname, "dist");
  const isProduction =
    process.env.NODE_ENV === "production" ||
    (process.env.NODE_ENV !== "development" && fs.existsSync(path.join(distDir, "index.html")));

  if (!isProduction) {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(distDir));
    app.use((req: Request, res: Response, next) => {
      if (req.method !== "GET" || req.path.startsWith("/api/")) {
        next();
        return;
      }
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  app.listen(port, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
