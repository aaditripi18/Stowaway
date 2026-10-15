// Python's repr() for a string, for messages that mirror f"...{x!r}..." in the
// Python package. Python prefers single quotes, switching to double only when the
// string contains a single quote but no double quote.
export function pyRepr(s: string): string {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const code = ch.codePointAt(0)!;
    if (ch === "\\" || ch === quote) out += "\\" + ch;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (code < 0x20 || code === 0x7f) out += "\\x" + code.toString(16).padStart(2, "0");
    else out += ch;
  }
  return out + quote;
}

// Python's str() for a float embedded in a message: integer-valued floats keep
// a trailing ".0" (str(0.0) == "0.0"), unlike canon() which strips it in detail.
export function pyStr(x: number): string {
  return Number.isInteger(x) ? x.toFixed(1) : String(x);
}
