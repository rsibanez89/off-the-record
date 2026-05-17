import { MODELS, modelLabel, type ModelId, type ModelPanelRole } from '../lib/audio';

export interface ModelPickerProps {
  value: ModelId;
  onChange: (id: ModelId) => void;
  disabled?: boolean;
  /** Determines which label variant is shown ("recommended for live" vs batch). */
  role: ModelPanelRole;
}

export function ModelPicker({ value, onChange, disabled, role }: ModelPickerProps) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as ModelId)}
      className="px-3 py-2 rounded border border-neutral-300 bg-white text-sm disabled:opacity-50"
    >
      {MODELS.map((m) => (
        <option key={m.id} value={m.id}>
          {modelLabel(m.id, role)}
        </option>
      ))}
    </select>
  );
}
