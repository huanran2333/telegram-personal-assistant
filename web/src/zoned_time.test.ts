import { describe, expect, it } from 'vitest';
import { zonedInput, zonedISO } from './zoned_time';

describe('Calendar approval time identity', () => {
  it('keeps the configured zone offset instead of silently converting to UTC Z', () => {
    expect(zonedISO('2026-10-12T16:00', 'Asia/Shanghai')).toBe('2026-10-12T16:00:00+08:00');
    expect(zonedInput('2026-10-12T08:00:00Z', 'Asia/Shanghai')).toBe('2026-10-12T16:00');
  });
  it('fails closed for DST gaps and ambiguous approved wall times', () => {
    expect(() => zonedISO('2026-03-08T02:30', 'America/New_York')).toThrow('CalendarTimeDoesNotExist');
    expect(() => zonedISO('2026-11-01T01:30', 'America/New_York')).toThrow('AmbiguousCalendarTime');
  });
});
