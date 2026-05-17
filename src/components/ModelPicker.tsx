import { modelLabel, modelsForRole, type ModelId, type ModelPanelRole } from '../lib/audio';

export interface ModelPickerProps {
  value: ModelId;
  onChange: (id: ModelId) => void;
  disabled?: boolean;
  /**
   * Determines which label variant is shown ("recommended for live" vs batch)
   * and which models are offered. Live hides models that lack word-level
   * timestamp support because LA-2 requires per-word timing.
   */
  role: ModelPanelRole;
}

export function ModelPicker({ value, onChange, disabled, role }: ModelPickerProps) {
  const models = modelsForRole(role);
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as ModelId)}
      className="px-3 py-2 rounded border border-neutral-300 bg-white text-sm disabled:opacity-50"
    >
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {modelLabel(m.id, role)}
        </option>
      ))}
    </select>
  );
}
