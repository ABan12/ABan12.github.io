#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/comments-test-override.yml"
tmp_site="${tmp_dir}/site"
tmp_giscus_post="_posts/2000-01-01-comments-integration-giscus.md"
tmp_disqus_post="_posts/2000-01-02-comments-integration-disqus.md"

cleanup() {
  rm -f "${tmp_giscus_post}" "${tmp_disqus_post}"
  rmdir _posts 2>/dev/null || true
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p _posts

cat >"${tmp_giscus_post}" <<'MARKDOWN'
---
layout: post
title: giscus comments integration fixture
date: 2000-01-01
permalink: /blog/comments-integration-giscus/
giscus_comments: true
related_posts: false
---

Temporary integration fixture.
MARKDOWN

cat >"${tmp_disqus_post}" <<'MARKDOWN'
---
layout: post
title: disqus comments integration fixture
date: 2000-01-02
permalink: /blog/comments-integration-disqus/
disqus_comments: true
related_posts: false
---

Temporary integration fixture.
MARKDOWN

cat >"${tmp_override}" <<'YAML'
giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

bundle exec jekyll build --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

giscus_page="${tmp_site}/blog/comments-integration-giscus/index.html"
disqus_page="${tmp_site}/blog/comments-integration-disqus/index.html"

grep -q 'https://giscus.app/client.js' "${giscus_page}"
if grep -q 'giscus comments misconfigured' "${giscus_page}"; then
  echo "unexpected giscus misconfiguration warning in ${giscus_page}" >&2
  exit 1
fi

grep -q 'id="disqus_thread"' "${disqus_page}"
grep -q '.disqus.com/embed.js' "${disqus_page}"

echo "comments integration checks passed"
