/** Convert a wall-clock form value in the user's configured zone to an ISO
 * instant carrying that zone's actual offset. Reject DST gaps and overlaps
 * instead of silently moving an approved Calendar event. */
function parts(instant: number, zone: string): number[] {
  const values = new Intl.DateTimeFormat('en-US', {
    timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(new Date(instant));
  const get = (name: string) => Number(values.find(part => part.type === name)?.value);
  return [get('year'), get('month'), get('day'), get('hour'), get('minute')];
}

export function zonedInput(instant: string, zone: string): string {
  const values = parts(new Date(instant).getTime(), zone);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${values[0]}-${pad(values[1])}-${pad(values[2])}T${pad(values[3])}:${pad(values[4])}`;
}

export function zonedISO(wall: string, zone: string): string {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(wall)) throw new Error('InvalidCalendarTime');
  const [year, month, day, hour, minute] = wall.match(/\d+/g)!.map(Number);
  const naive = Date.UTC(year, month - 1, day, hour, minute);
  const requested = [year, month, day, hour, minute];
  // Offsets on both sides of the day cover normal, one-hour and half-hour
  // DST changes. A wall time must resolve to precisely one instant.
  const offsets = new Set([-86400000, 0, 86400000].map(delta => {
    const sample = naive + delta;
    const [y, m, d, h, min] = parts(sample, zone);
    return Date.UTC(y, m - 1, d, h, min) - sample;
  }));
  const matches = [...offsets].map(offset => naive - offset).filter(candidate =>
    parts(candidate, zone).every((value, i) => value === requested[i]));
  if (matches.length === 0) throw new Error('CalendarTimeDoesNotExist');
  if (matches.length !== 1) throw new Error('AmbiguousCalendarTime');
  const instant = matches[0];
  const offsetMinutes = (naive - instant) / 60000;
  const sign = offsetMinutes < 0 ? '-' : '+';
  const absolute = Math.abs(offsetMinutes);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${wall}:00${sign}${pad(Math.floor(absolute / 60))}:${pad(absolute % 60)}`;
}
