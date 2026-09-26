# Personal AI Research Website

This is an Astro + Markdown-style personal research site designed to deploy on GitHub Pages.

## 1. Install Node.js on Ubuntu

If `node` is still missing:

```bash
sudo apt update
sudo apt install -y nodejs npm
node --version
npm --version
```

For a newer Node version, use nvm instead.

## 2. Run locally

```bash
cd shubho-ai-site
npm install
npm run dev
```

Open the local URL printed by Astro.

## 3. Personalize

Search for and replace:

- `Your Name`
- `YOUR-USERNAME`
- `YOUR-EMAIL@example.com`
- `https://YOUR-USERNAME.github.io`

Put your CV at:

```text
public/resume.pdf
```

## 4. Create the GitHub repository

For the cleanest URL, create a repository named:

```text
YOUR-USERNAME.github.io
```

Then:

```bash
git init
git add .
git commit -m "Initial personal research site"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-USERNAME.github.io.git
git push -u origin main
```

Because the repository has the special `username.github.io` name, no Astro `base` path is needed.

## 5. Enable GitHub Pages

GitHub → repository → Settings → Pages → Source:

**GitHub Actions**

Every push to `main` will rebuild and deploy the site.

## 6. Custom domain

If you later buy a domain, add a `public/CNAME` file containing the domain and update `site` in `astro.config.mjs`. Then configure the DNS records at your domain provider.

## Writing

The current scaffold keeps writing intentionally simple. Add future posts as Markdown/content entries and link them from `src/pages/writing/index.astro`.

## Important

The site currently contains placeholder identity/contact information. Replace those before publishing.
