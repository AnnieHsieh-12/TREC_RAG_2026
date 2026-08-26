export function defaultCaseFold(
  input: string,
  mappings: ReadonlyMap<number, readonly number[]>,
): string {
  const output: string[] = [];

  for (const character of input) {
    const codePoint = character.codePointAt(0);
    if (codePoint === undefined) {
      continue;
    }

    const mappedCodePoints = mappings.get(codePoint);
    if (mappedCodePoints === undefined) {
      output.push(character);
      continue;
    }

    output.push(String.fromCodePoint(...mappedCodePoints));
  }

  return output.join("");
}
