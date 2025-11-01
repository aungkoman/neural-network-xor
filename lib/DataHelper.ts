// A basic set of Myanmar characters. A more comprehensive set would be needed for a real-world app.
export const MYANMAR_CHARS = 'ကခဂဃငစဆဇဈညတထဒဓနပဖဗဘမယရလဝသဟဠအာိီုူေဲံ့း'.split('');
export const VOCAB_SIZE = MYANMAR_CHARS.length;

export const GENDERS = ['male', 'female'];

export interface NameData {
  name: string;
  gender: 'male' | 'female';
}

// A small, curated dataset for demonstration purposes.
export const nameDataset: NameData[] = [
  { name: 'အောင်အောင်', gender: 'male' },
  { name: 'ကျော်စွာ', gender: 'male' },
  { name: 'မင်းသူ', gender: 'male' },
  { name: 'ကိုကို', gender: 'male' },
  { name: 'ဇော်ဇော်', gender: 'male' },
  { name: 'သူရ', gender: 'male' },
  { name: 'နေမျိုး', gender: 'male' },
  { name: 'စည်သူ', gender: 'male' },
  
  { name: 'စုစု', gender: 'female' },
  { name: 'ခင်ခင်', gender: 'female' },
  { name: 'မေသူ', gender: 'female' },
  { name: 'နွယ်နွယ်', gender: 'female' },
  { name: 'စန္ဒာ', gender: 'female' },
  { name: 'မိုးမိုး', gender: 'female' },
  { name: 'အေးအေး', gender: 'female' },
  { name: 'သီတာ', gender: 'female' },
];

/**
 * Converts a Myanmar name into a multi-hot encoded vector.
 * The vector has a length equal to the vocabulary size.
 * A '1' at an index indicates the presence of the character in the name.
 * @param name The name to encode.
 * @returns A numerical vector.
 */
export function nameToVector(name: string): number[] {
  const vector = new Array(VOCAB_SIZE).fill(0);
  for (const char of name) {
    const index = MYANMAR_CHARS.indexOf(char);
    if (index !== -1) {
      vector[index] = 1;
    }
  }
  return vector;
}

/**
 * Converts a gender string to a one-hot encoded vector.
 * 'male' -> [1, 0]
 * 'female' -> [0, 1]
 * @param gender The gender string.
 * @returns A one-hot encoded vector.
 */
export function genderToVector(gender: 'male' | 'female'): number[] {
  return gender === 'male' ? [1, 0] : [0, 1];
}

/**
 * Converts the network's output vector to a predicted gender and confidence.
 * @param outputVector The output from the neural network (e.g., [0.9, 0.1]).
 * @returns An object with the predicted gender and confidence score.
 */
export function vectorToPrediction(outputVector: number[]): { gender: string, confidence: number } {
  const maleConfidence = outputVector[0];
  const femaleConfidence = outputVector[1];
  
  if (maleConfidence > femaleConfidence) {
    return { gender: 'male', confidence: maleConfidence };
  } else {
    return { gender: 'female', confidence: femaleConfidence };
  }
}
