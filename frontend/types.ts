export interface SourceResult {
  work_title: string;
  work_slug: string;
  section_number: number | null;
  chunk_type: string;
  content: string;
  similarity: number;
  used: boolean;
}

export interface QueryResponse {
  answer: string;
  sources: SourceResult[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceResult[];
}

export type Period = "early" | "middle" | "late" | null;

export const WORKS: { label: string; slug: string }[] = [
  { label: "Beyond Good and Evil", slug: "beyond_good_and_evil" },
  { label: "On the Genealogy of Morality", slug: "genealogy_of_morality" },
  { label: "Twilight of the Idols", slug: "twilight_of_the_idols" },
  { label: "The Antichrist", slug: "the_antichrist" },
  { label: "Ecce Homo", slug: "ecce_homo" },
  { label: "Nietzsche contra Wagner", slug: "nietzsche_contra_wagner" },
  { label: "The Gay Science", slug: "the_gay_science" },
  { label: "Dawn", slug: "dawn" },
  { label: "Human, All Too Human", slug: "human_all_too_human" },
  { label: "Thus Spoke Zarathustra", slug: "thus_spoke_zarathustra" },
  { label: "The Birth of Tragedy", slug: "birth_of_tragedy" },
  { label: "Untimely Meditations", slug: "untimely_meditations" },
];
