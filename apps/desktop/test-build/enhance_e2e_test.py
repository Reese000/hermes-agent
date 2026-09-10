"""
E2E QA test: Verify Enhance button replaces composer text.
Connects to isolated Hermes Desktop via CDP on port 9334.
"""
import asyncio
import json
import time
from playwright.async_api import async_playwright

async def main():
    print("=== Enhance Button E2E Test ===")
    print(f"CDP target: http://127.0.0.1:9334")
    
    async with async_playwright() as p:
        # Connect to isolated instance via CDP
        browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9334")
        print(f"Connected to browser: {browser.version}")
        
        # Get the main page
        contexts = browser.contexts
        print(f"Contexts: {len(contexts)}")
        
        page = None
        for ctx in contexts:
            for pg in ctx.pages:
                if "index.html" in pg.url:
                    page = pg
                    break
            if page:
                break
        
        if not page:
            # Try first page
            if contexts and contexts[0].pages:
                page = contexts[0].pages[0]
        
        if not page:
            print("ERROR: No page found!")
            return
        
        print(f"Page URL: {page.url}")
        print(f"Page title: {await page.title()}")
        
        # Wait for app to fully load
        print("Waiting for app to load...")
        await page.wait_for_load_state("networkidle", timeout=15000)
        await asyncio.sleep(3)
        
        # Check if we're on setup screen or main chat
        print("Checking current state...")
        
        # Look for the composer/textarea
        composer = None
        selectors = [
            'textarea[placeholder*="message"]',
            'textarea[placeholder*="Message"]',
            'textarea[placeholder*="Ask"]',
            'textarea',
            '[contenteditable="true"]',
            'div[class*="composer"] textarea',
            'div[class*="composer"] [contenteditable]',
        ]
        
        for sel in selectors:
            try:
                el = await page.wait_for_selector(sel, timeout=3000)
                if el:
                    composer = el
                    print(f"Found composer with selector: {sel}")
                    break
            except:
                continue
        
        if not composer:
            # Try to find via JavaScript
            print("Trying JavaScript to find composer...")
            result = await page.evaluate("""() => {
                const textareas = document.querySelectorAll('textarea');
                const editables = document.querySelectorAll('[contenteditable="true"]');
                return {
                    textareas: textareas.length,
                    editables: editables.length,
                    textareaPlaceholders: Array.from(textareas).map(t => t.placeholder),
                    editableTexts: Array.from(editables).map(e => e.textContent?.substring(0, 50))
                };
            }""")
            print(f"JS search result: {json.dumps(result, indent=2)}")
            
            if result['textareas'] > 0:
                composer = await page.query_selector('textarea')
            elif result['editables'] > 0:
                composer = await page.query_selector('[contenteditable="true"]')
        
        if not composer:
            print("ERROR: Could not find composer element!")
            # Take screenshot for debugging
            await page.screenshot(path="debug-no-composer.png")
            print("Screenshot saved: debug-no-composer.png")
            return
        
        # Set test text
        test_prompt = "write a simple hello world function in python"
        print(f"Setting composer text: '{test_prompt}'")
        
        # Clear and type
        await composer.click()
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Delete")
        await asyncio.sleep(0.5)
        
        # Check if it's a textarea or contenteditable
        tag = await composer.evaluate("el => el.tagName.toLowerCase()")
        print(f"Composer tag: {tag}")
        
        if tag == "textarea":
            await composer.fill(test_prompt)
        else:
            # contenteditable
            await composer.evaluate(f"el => el.textContent = '{test_prompt}'")
            await page.keyboard.press("Enter")
        
        await asyncio.sleep(1)
        
        # Read back the pre-enhance text
        pre_text = await composer.evaluate("el => el.value || el.textContent || ''")
        print(f"PRE-ENHANCE text: '{pre_text}'")
        
        # Take pre-click screenshot
        await page.screenshot(path="pre-enhance.png")
        print("Screenshot saved: pre-enhance.png")
        
        # Find the Enhance button
        print("Looking for Enhance button...")
        enhance_btn = None
        
        # Try various selectors for the enhance button
        enhance_selectors = [
            'button:has-text("Enhance")',
            '[aria-label*="Enhance"]',
            '[title*="Enhance"]',
            'button[class*="enhance"]',
            'div[class*="enhance"]',
        ]
        
        for sel in enhance_selectors:
            try:
                el = await page.wait_for_selector(sel, timeout=3000)
                if el:
                    enhance_btn = el
                    print(f"Found Enhance button with selector: {sel}")
                    break
            except:
                continue
        
        if not enhance_btn:
            # Try JavaScript
            print("Trying JavaScript to find Enhance button...")
            result = await page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                const enhanceButtons = [];
                for (const btn of buttons) {
                    const text = btn.textContent || '';
                    const aria = btn.getAttribute('aria-label') || '';
                    const title = btn.getAttribute('title') || '';
                    if (text.toLowerCase().includes('enhance') || 
                        aria.toLowerCase().includes('enhance') ||
                        title.toLowerCase().includes('enhance')) {
                        enhanceButtons.push({
                            text: text.substring(0, 50),
                            aria: aria,
                            title: title,
                            disabled: btn.disabled
                        });
                    }
                }
                return enhanceButtons;
            }""")
            print(f"JS Enhance search: {json.dumps(result, indent=2)}")
            
            if result:
                # Try to find by text content
                enhance_btn = await page.query_selector(f'button:has-text("Enhance")')
        
        if not enhance_btn:
            print("ERROR: Could not find Enhance button!")
            await page.screenshot(path="debug-no-enhance.png")
            print("Screenshot saved: debug-no-enhance.png")
            
            # Dump all buttons for debugging
            buttons = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('button')).map(b => ({
                    text: (b.textContent || '').substring(0, 50),
                    aria: b.getAttribute('aria-label') || '',
                    title: b.getAttribute('title') || '',
                    disabled: b.disabled,
                    classes: b.className.substring(0, 100)
                }));
            }""")
            print(f"All buttons: {json.dumps(buttons, indent=2)}")
            return
        
        # Click Enhance
        print("Clicking Enhance button...")
        await enhance_btn.click()
        
        # Wait for network activity / toast
        print("Waiting for enhancement to complete...")
        start = time.time()
        max_wait = 30
        
        # Monitor for network requests
        network_requests = []
        def on_request(request):
            if "enhance" in request.url.lower():
                network_requests.append({
                    "url": request.url,
                    "method": request.method,
                    "time": time.time()
                })
        
        network_responses = []
        def on_response(response):
            if "enhance" in response.url.lower():
                network_responses.append({
                    "url": response.url,
                    "status": response.status,
                    "time": time.time()
                })
        
        page.on("request", on_request)
        page.on("response", on_response)
        
        # Wait up to max_wait seconds
        while time.time() - start < max_wait:
            await asyncio.sleep(1)
            
            # Check for toast/notification
            toasts = await page.evaluate("""() => {
                const notifications = document.querySelectorAll('[class*="toast"], [class*="notification"], [class*="snack"], [role="alert"]');
                return Array.from(notifications).map(n => ({
                    text: n.textContent?.substring(0, 100),
                    classes: n.className?.substring(0, 100)
                }));
            }""")
            
            if toasts:
                print(f"Toast detected: {json.dumps(toasts)}")
                # Check if enhance failed toast appeared
                for toast in toasts:
                    if "fail" in (toast.get("text", "") or "").lower():
                        print("Enhance FAILED toast detected!")
                        break
                break
            
            # Check if composer text changed
            current_text = await composer.evaluate("el => el.value || el.textContent || ''")
            if current_text != pre_text:
                print(f"Composer text changed! New length: {len(current_text)}")
                break
            
            elapsed = time.time() - start
            if elapsed > 5 and not network_requests:
                print(f"No enhance request sent after {elapsed:.1f}s")
        
        # Read post-enhance text
        post_text = await composer.evaluate("el => el.value || el.textContent || ''")
        print(f"POST-ENHANCE text: '{post_text}'")
        
        # Take post-click screenshot
        await page.screenshot(path="post-enhance.png")
        print("Screenshot saved: post-enhance.png")
        
        # Remove listeners
        page.remove_listener("request", on_request)
        page.remove_listener("response", on_response)
        
        # Print network evidence
        print(f"\n=== Network Evidence ===")
        print(f"Enhance requests: {len(network_requests)}")
        for req in network_requests:
            print(f"  {req['method']} {req['url']}")
        print(f"Enhance responses: {len(network_responses)}")
        for resp in network_responses:
            print(f"  Status {resp['status']}: {resp['url']}")
        
        # Final verdict
        print(f"\n=== VERDICT ===")
        print(f"Pre-enhance:  '{pre_text}'")
        print(f"Post-enhance: '{post_text}'")
        
        if post_text != pre_text and len(post_text) > 0:
            print("PASS: Enhance button successfully changed composer text!")
        elif post_text == pre_text:
            print("FAIL: Composer text unchanged after clicking Enhance")
            if not network_requests:
                print("  No enhance request was sent - click may not have reached the app")
        else:
            print("FAIL: Post-enhance text is empty")
        
        # Close browser (don't kill the app)
        await browser.close()
        print("\nBrowser disconnected (app still running)")

asyncio.run(main())
