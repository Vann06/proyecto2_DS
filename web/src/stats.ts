// Cálculos pequeños que se hacen en el navegador al cambiar un filtro.

export function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function quantile(sorted: number[], q: number): number {
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

export function pearson(x: number[], y: number[]): number {
  const mx = mean(x);
  const my = mean(y);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < x.length; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
}

export function rmse(real: number[], predicted: number[]): number {
  return Math.sqrt(mean(real.map((value, i) => (value - predicted[i]) ** 2)));
}

export function r2(real: number[], predicted: number[]): number {
  const m = mean(real);
  const residual = real.reduce((sum, value, i) => sum + (value - predicted[i]) ** 2, 0);
  const total = real.reduce((sum, value) => sum + (value - m) ** 2, 0);
  return 1 - residual / total;
}

// Promedio de y en grupos con el mismo número de puntos ordenados por x: la tendencia sin suponer una forma
export function binnedMeans(x: number[], y: number[], bins = 20): [number, number][] {
  const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
  const size = Math.ceil(order.length / bins);
  const result: [number, number][] = [];
  for (let start = 0; start < order.length; start += size) {
    const chunk = order.slice(start, start + size);
    result.push([mean(chunk.map((i) => x[i])), mean(chunk.map((i) => y[i]))]);
  }
  return result;
}

export function boxStats(values: number[]): [number, number, number, number, number] {
  const sorted = [...values].sort((a, b) => a - b);
  return [quantile(sorted, 0.05), quantile(sorted, 0.25), quantile(sorted, 0.5), quantile(sorted, 0.75),
    quantile(sorted, 0.95)];
}

export function histogram(values: number[], start: number, end: number, bins: number): number[] {
  const counts = new Array(bins).fill(0);
  const width = (end - start) / bins;
  for (const value of values) {
    counts[Math.min(bins - 1, Math.max(0, Math.floor((value - start) / width)))] += 1;
  }
  return counts;
}

export function signed(value: number, digits = 2): string {
  const text = Math.abs(value).toFixed(digits);
  // Un valor que se redondea a cero no lleva signo ("0.00" y no "−0.00")
  if (Number(text) === 0) return text;
  return `${value > 0 ? "+" : "−"}${text}`;
}

export const fixed = (value: number, digits = 3) => value.toFixed(digits);
export const integer = (value: number) => Math.round(value).toLocaleString("en-US");
