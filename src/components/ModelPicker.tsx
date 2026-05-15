import { MODELS, type ModelId } from '../lib/audio';

export interface ModelPickerProps {
  value: ModelId;
  onChange: (id: ModelId) => void;
  disabled?: boolean;
}

export function ModelPicker({ value, onChange, disabled }: ModelPickerProps) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as ModelId)}
      className="px-3 py-2 rounded border border-neutral-300 bg-white text-sm disabled:opacity-50"
    >
      {MODELS.map((m) => (
        <option key={m.id} value={m.id}>
          {m.label}
        </option>
      ))}
    </select>
  );
}
