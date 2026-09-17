export default [
  {
    files: ["frontend/**/*.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "script",
      globals: {
        document: "readonly",
        fetch: "readonly",
        FormData: "readonly",
        Math: "readonly",
        Number: "readonly",
        Object: "readonly",
        Promise: "readonly",
        Set: "readonly",
        String: "readonly",
        Date: "readonly",
        window: "readonly",
        performance: "readonly",
      },
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["warn", { "args": "none" }],
    },
  },
];
