import { QuizData } from '../types';

/**
 * Extrage textul util dintr-un fișier HTML, eliminând tag-urile și script-urile
 */
function extractTextFromHTML(html: string): string {
  // Elimină tag-urile <script> și <style> cu tot conținutul
  let text = html.replace(/<script[\s\S]*?<\/script>/gi, '');
  text = text.replace(/<style[\s\S]*?<\/style>/gi, '');
  text = text.replace(/<!--[\s\S]*?-->/g, '');

  // Elimină tag-urile HTML
  text = text.replace(/<[^>]+>/g, ' ');

  // Decodifică entități HTML comune
  text = text.replace(/&nbsp;/g, ' ');
  text = text.replace(/&amp;/g, '&');
  text = text.replace(/&lt;/g, '<');
  text = text.replace(/&gt;/g, '>');
  text = text.replace(/&quot;/g, '"');
  text = text.replace(/&#39;/g, "'");
  text = text.replace(/&ndash;/g, '-');
  text = text.replace(/&mdash;/g, '-');

  // Normalizare whitespace
  text = text.replace(/\s+/g, ' ').trim();

  return text;
}

export async function generateQuizWithAI(
  topic: string,
  difficulty: 'easy' | 'medium' | 'hard',
  rounds: number,
  sourceContent?: string
): Promise<QuizData> {
  const response = await fetch('/api/gemini/generate-quiz', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      topic,
      difficulty,
      rounds,
      sourceContent,
    }),
  });

  if (!response.ok) {
    let errorMsg = 'Eroare la apelarea serviciului AI';
    const rawText = await response.text();
    if (rawText) {
      try {
        const errorData = JSON.parse(rawText);
        if (errorData?.error) {
          errorMsg = errorData.error;
        } else {
          errorMsg = rawText;
        }
      } catch {
        errorMsg = rawText;
      }
    }
    throw new Error(errorMsg);
  }

  return (await response.json()) as QuizData;
}

/**
 * Utilitar pentru extragerea textului din HTML (exportat pentru a fi folosit în Dashboard)
 */
export { extractTextFromHTML };
