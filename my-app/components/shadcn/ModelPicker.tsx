"use client";
import Image from "next/image";
import type { FC } from "react";
import openai from "../../assets/providers/openai.svg";
import google from "../../assets/providers/google.svg";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

const models = [
  {
    name: "GPT-4o",
    value: "gpt-4o",
    icon: openai,
  },
  {
    name: "GPT-4o mini",
    value: "gpt-4o-mini",
    icon: openai,
  },
  {
    name: "My FastAPI Gemini",
    value: "my-fastapi",
    icon: google,
  },
];

type ModelPickerProps = {
  value: string;
  onChange: (value: string) => void;
};

export const ModelPicker: FC<ModelPickerProps> = ({ value, onChange }) => {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="max-w-[300px]">
        <SelectValue placeholder="Select model" />
      </SelectTrigger>
      <SelectContent className="">
        {models.map((model) => (
          <SelectItem key={model.value} value={model.value}>
            <span className="flex items-center gap-2">
              <Image
                src={model.icon}
                alt={model.name}
                className="inline size-4"
              />
              <span>{model.name}</span>
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
