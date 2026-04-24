/**
 * Per-product defect taxonomies.
 *
 * Canonical MVTec defect class names per category, plus Decospan (Dutch).
 * Operators can extend these at runtime via POST /api/taxonomy/{product}.
 *
 * Colour palette cycles through 7 distinct hues. Shortcuts are the digits
 * "1".."9" by array index; taxonomies longer than 9 entries drop shortcut
 * on the tail (rare — only `cable` at 8 falls near the limit).
 */

export interface TaxonomyEntry {
  key: string;          // machine identifier (snake_case)
  name: string;         // display name
  color: string;        // hex colour
  shortcut: string;     // '1'..'9' or ''
  custom?: boolean;     // true for operator-added categories
}

const PALETTE = [
  '#EF4444', // red
  '#F97316', // orange
  '#EAB308', // yellow
  '#22C55E', // green
  '#06B6D4', // cyan
  '#A855F7', // purple
  '#EC4899', // pink
];

function buildTaxonomy(keys: string[], displayOverrides: Record<string, string> = {}): TaxonomyEntry[] {
  return keys.map((key, i) => ({
    key,
    name: displayOverrides[key] ?? key.replace(/_/g, ' '),
    color: PALETTE[i % PALETTE.length],
    shortcut: i < 9 ? String(i + 1) : '',
  }));
}

const MVTEC_CATEGORIES: Record<string, string[]> = {
  bottle:     ['broken_large', 'broken_small', 'contamination'],
  cable:      ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation',
               'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation'],
  capsule:    ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
  carpet:     ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
  grid:       ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
  hazelnut:   ['crack', 'cut', 'hole', 'print'],
  leather:    ['color', 'cut', 'fold', 'glue', 'poke'],
  metal_nut:  ['bent', 'color', 'flip', 'scratch'],
  pill:       ['color', 'combined', 'contamination', 'crack', 'faulty_imprint',
               'pill_type', 'scratch'],
  screw:      ['manipulated_front', 'scratch_head', 'scratch_neck',
               'thread_side', 'thread_top'],
  tile:       ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
  toothbrush: ['defective'],
  transistor: ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
  wood:       ['color', 'combined', 'hole', 'liquid', 'scratch'],
  zipper:     ['broken_teeth', 'combined', 'fabric_border', 'fabric_interior',
               'rough', 'split_teeth', 'squeezed_teeth'],
};

const DECOSPAN_CATEGORIES = [
  'krassen', 'deuk', 'vlekken', 'barst', 'open_fout', 'open_knop', 'snijfout',
];

const DECOSPAN_DISPLAY: Record<string, string> = {
  krassen:   'krassen (scratches)',
  deuk:      'deuk (dent)',
  vlekken:   'vlekken (stains)',
  barst:     'barst (crack)',
  open_fout: 'open fout (open defect)',
  open_knop: 'open knop (open knot)',
  snijfout:  'snijfout (cutting error)',
};

export const BASE_TAXONOMIES: Record<string, TaxonomyEntry[]> = {
  ...Object.fromEntries(
    Object.entries(MVTEC_CATEGORIES).map(([product, keys]) => [
      product,
      buildTaxonomy(keys),
    ]),
  ),
  decospan: buildTaxonomy(DECOSPAN_CATEGORIES, DECOSPAN_DISPLAY),
};

/** Fallback palette used when the product has no base taxonomy. */
export const GENERIC_FALLBACK: TaxonomyEntry[] = [
  { key: 'anomaly',        name: 'anomaly',        color: PALETTE[0], shortcut: '1' },
  { key: 'contamination',  name: 'contamination',  color: PALETTE[1], shortcut: '2' },
  { key: 'structural',     name: 'structural',     color: PALETTE[2], shortcut: '3' },
];

export function getBaseTaxonomy(product_class: string | null | undefined): TaxonomyEntry[] {
  if (!product_class) return GENERIC_FALLBACK;
  return BASE_TAXONOMIES[product_class] ?? GENERIC_FALLBACK;
}

/** Merge base taxonomy with operator-added custom categories, de-duping by key. */
export function mergeTaxonomy(
  base: TaxonomyEntry[],
  custom: TaxonomyEntry[],
): TaxonomyEntry[] {
  const seen = new Set(base.map(e => e.key.toLowerCase()));
  const extras = custom.filter(e => !seen.has(e.key.toLowerCase()));
  return [...base, ...extras.map(e => ({ ...e, custom: true }))];
}

/** Next unused palette colour (cycles if all used). */
export function nextColor(existing: TaxonomyEntry[]): string {
  const used = new Set(existing.map(e => e.color));
  for (const c of PALETTE) {
    if (!used.has(c)) return c;
  }
  return PALETTE[existing.length % PALETTE.length];
}

/** Next unused digit shortcut in '1'..'9' — returns '' if all taken. */
export function nextShortcut(existing: TaxonomyEntry[]): string {
  const used = new Set(existing.map(e => e.shortcut).filter(Boolean));
  for (let i = 1; i <= 9; i++) {
    const s = String(i);
    if (!used.has(s)) return s;
  }
  return '';
}

/** Validate a proposed custom-category key. Returns error message or null. */
export function validateCategoryKey(
  key: string,
  existing: TaxonomyEntry[],
): string | null {
  const k = key.trim();
  if (k.length < 2 || k.length > 30) return 'must be 2–30 characters';
  if (!/^[a-z0-9_]+$/.test(k)) return 'use lowercase letters, digits, or _';
  const lower = k.toLowerCase();
  if (existing.some(e => e.key.toLowerCase() === lower)) {
    return 'already exists for this product';
  }
  return null;
}

export function toSnakeCase(raw: string): string {
  return raw
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}
