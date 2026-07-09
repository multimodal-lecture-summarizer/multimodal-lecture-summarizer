export function parseUTCDate(dateString: string | Date | null | undefined): Date | null {
  if (!dateString) return null;
  if (dateString instanceof Date) return dateString;
  
  // If string already contains Z or +/-, it has timezone info
  if (dateString.includes('Z') || dateString.match(/[+-]\d{2}:\d{2}$/)) {
    return new Date(dateString);
  }
  
  // Append Z to force UTC parsing
  return new Date(dateString + 'Z');
}

export function formatUTCDate(dateString: string | null | undefined, locale = 'vi-VN'): string {
  const date = parseUTCDate(dateString);
  if (!date) return '-';
  return date.toLocaleDateString(locale);
}

export function formatUTCTime(dateString: string | null | undefined, locale = 'vi-VN'): string {
  const date = parseUTCDate(dateString);
  if (!date) return '-';
  return date.toLocaleTimeString(locale);
}

export function formatUTCDateString(dateString: string | null | undefined, locale = 'vi-VN', options?: Intl.DateTimeFormatOptions): string {
  const date = parseUTCDate(dateString);
  if (!date) return '-';
  return date.toLocaleDateString(locale, options);
}

export function formatUTCDateTime(dateString: string | null | undefined, locale = 'vi-VN'): string {
  const date = parseUTCDate(dateString);
  if (!date) return '-';
  return date.toLocaleString(locale);
}
